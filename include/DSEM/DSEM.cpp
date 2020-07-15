/*
 * Copyright (C) 2020 Dillon Cislo
 *
 * This file is part of DSEM++.
 *
 * DSEM++ is free software: you can redistribute it and/or modify it under the terms
 * of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will by useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program.
 * If not, see <http://www.gnu.org/licenses/>
 *
 */

#include "DSEM.h"
#include "clipToUnitCircle.h"
#include "../LBFGS/BFGSMat.h"
#include "../LBFGS/LineSearchBacktracking.h"

#include <cassert>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <omp.h>

#include <igl/edges.h>
#include <igl/internal_angles.h>
#include <igl/boundary_loop.h>
#include <igl/setdiff.h>
#include <igl/edge_lengths.h>
#include <igl/doublearea.h>
#include <igl/unique_edge_map.h>

// ======================================================================================
// CONSTRUCTOR FUNCTIONS
// ======================================================================================

///
/// Default constructor
///
template <typename Scalar, typename Index>
DSEMpp::DSEM<Scalar, Index>::DSEM(
    const Eigen::Matrix<Index, Eigen::Dynamic, 3> &F,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &V,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 2> &x,
    const DSEMParam<Scalar> &dsemParam,
    const NNIpp::NNIParam<Scalar> &nniParam ) : m_F(F), m_V(V), m_x(x) {

  // Verify input triangulation entries
  if ( ( m_F.array().isNaN().any() ) || ( m_F.array().isInf().any() ) ) {
    std::runtime_error("Invalid face connectivity list");
  }

  if ( ( m_V.array().isNaN().any() ) || ( m_V.array().isInf().any() ) ) {
    std::runtime_error("Invalid 3D vertex coordinate list");
  }

  if ( ( m_x.array().isNaN().any() ) || ( m_x.array().isInf().any() ) ) {
    std::runtime_error("Invalid 2D vertex coordinate list");
  }

  if ( m_V.rows() != m_x.rows() ) {
    std::runtime_error("Inconsistent number of vertices");
  }

  // Construct the triangulation edge list
  Eigen::Matrix<Index, Eigen::Dynamic, 2> dE(1,2); // Directed edge list
  m_E = Eigen::Matrix<Index, Eigen::Dynamic, 2>::Zero(1,2); // Undirected edge list
  IndexVector FE(1);
  std::vector<std::vector<Index> > uE2E; // Unused - shouldn't be necessary...
  igl::unique_edge_map( m_F, dE, m_E, FE, uE2E );

  // Reserve space for the edge-face correspondence list
  m_EF.reserve( m_E.rows() );
  for( int i = 0; i < m_E.rows(); i++ ) {
    std::vector<Index> tmpVec;
    m_EF.push_back(tmpVec);
  }

  // Assemble the face-edge and edge-face correspondence lists
  m_FE.resizeLike( m_F );
  for( int i = 0; i < m_F.rows(); i++ ) {
    for( int j = 0; j < 3; j++ ) {

      m_FE(i,j) = FE( i + (j * m_F.rows()) );

      m_EF[ m_FE(i,j) ].push_back( (Index) i );

    }
  }

  // Verify that the triangulation is a topological disk
  int eulerChi = m_x.rows() - m_E.rows() + m_F.rows();
  if ( eulerChi != 1 ) {
    std::runtime_error("Input mesh must be a topological disk");
  }

  // Extract bulk/boundary vertices
  std::vector<std::vector<Index> > allBdyLoops;
  igl::boundary_loop( m_F, allBdyLoops );

  if ( allBdyLoops.size() != 1 ) {
    std::runtime_error("Input mesh has more than one boundary");
  }

  m_bdyIDx = IndexVector::Zero(allBdyLoops[0].size(), 1);
  igl::boundary_loop( m_F, m_bdyIDx );

  IndexVector allVIDx = IndexVector::LinSpaced(m_V.rows(), 0, (m_V.rows()-1));
  IndexVector IA(1, 1); // Not used - try to find way to eliminate
  m_bulkIDx = IndexVector::Zero(1,1);

  igl::setdiff( allVIDx, m_bdyIDx, m_bulkIDx, IA );

  // Clip the boundary vertices to the unit circle
  DSEMpp::clipToUnitCircle( m_bdyIDx, m_x );

  // Check that no points in the pullback mesh lie outside the unit disk
  if ( ( m_x.rowwise().norm().array() > Scalar(1.0) ).any() ) {
    std::runtime_error("Some pullback vertices lie outside the unit disk");
  }

  // Check for valid DSEM parameters
  dsemParam.checkParam();
  m_param = dsemParam;

  // Check for valid NNI parameters
  nniParam.checkParam();

  // Generate the surface interpolant
  Vector Xp = x.col(0);
  Vector Yp = x.col(1);
  Matrix Vp = V;
  NNIpp::NaturalNeighborInterpolant<Scalar> NNI( Xp, Yp, Vp, nniParam );
  m_NNI = NNI;

  // Construct the differential operators
  this->constructDifferentialOperators();

  // Construct averaging operators
  this->constructAveragingOperators();

  // Compute the areas of each face in the pullback mesh
  Vector faceAreas( m_F.rows(), 1 );
  igl::doublearea( m_x, m_F, faceAreas );
  faceAreas = (faceAreas.array() / Scalar(2.0)).matrix();

  // Compute the areas associated to each vertex in the pullback mesh
  m_vertexAreas = Vector::Zero( m_x.rows(), 1 );
  for( int i = 0; i < m_F.rows(); i++ ) {
    for( int j = 0; j < 3; j++ ) {

      m_vertexAreas( m_F(i,j) ) += faceAreas(i);

    }
  }

  m_vertexAreas = (m_vertexAreas.array() / Scalar(3.0)).matrix();

  // Construct the vertex area matrix
  RowVector VAT = m_vertexAreas.transpose();
  m_vertexAreaMat = VAT.replicate( m_x.rows(), 1 );

  // Construct the sparse identity matrix
  typedef Eigen::Triplet<Scalar> T;

  std::vector<T> tListI;
  tListI.reserve( 2 * m_V.rows() );

  for( int i = 0; i < (2 * m_V.rows()); i++ ) {
    tListI.push_back( T(i, i, Scalar(1.0)) );
  }

  Eigen::SparseMatrix<Scalar> speye( 2 * m_V.rows(), 2 * m_V.rows() );
  speye.setFromTriplets( tListI.begin(), tListI.end() );
  m_speye = speye;

};

///
/// Construct the intrinsic differential operators on the surface mesh
///
template <typename Scalar, typename Index>
void DSEMpp::DSEM<Scalar, Index>::constructDifferentialOperators() {

  // The number of vertices
  int numV = m_x.rows();

  // The number of faces
  int numF = m_F.rows();

  // A vector of face indices
  IndexVector u(numF, 1);
  u = IndexVector::LinSpaced(numF, 0, (numF-1));

  // Extract facet edges
  // NOTE: This expects that the faces are oriented CCW
  Eigen::Matrix<Scalar, Eigen::Dynamic, 2> e1(numF, 2);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 2> e2(numF, 2);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 2> e3(numF, 2);
  for( int i = 0; i < numF; i++ ) {

    e1.row(i) = m_x.row( m_F(i,2) ) - m_x.row( m_F(i,1) );
    e2.row(i) = m_x.row( m_F(i,0) ) - m_x.row( m_F(i,2) );
    e3.row(i) = m_x.row( m_F(i,1) ) - m_x.row( m_F(i,0) );

  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, 3> eX(numF, 3);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 3> eY(numF, 3);
  eX << e1.col(0), e2.col(0), e3.col(0);
  eY << e1.col(1), e2.col(1), e3.col(1);

  // Extract signed facet areas
  Vector a = ( e1.col(0).cwiseProduct( e2.col(1) )
      - e2.col(0).cwiseProduct( e1.col(1) ) ) / Scalar(2.0);

  // Construct sparse matrix triplets
  typedef Eigen::Triplet<Scalar> T;
  typedef Eigen::Triplet<CScalar> TC;

  std::vector<T> tListX, tListY;
  std::vector<TC> tListZ, tListC;

  tListX.reserve( 3 * numF );
  tListY.reserve( 3 * numF );
  tListZ.reserve( 3 * numF );
  tListC.reserve( 3 * numF );

  for( int i = 0; i < numF; i++ ) {
    for( int j = 0; j < 3; j++ ) {

      Scalar mx = -eY(i,j) / ( Scalar(2.0) * a(i) );
      Scalar my = eX(i,j) / ( Scalar(2.0) * a(i) );
      CScalar mz( mx / Scalar(2.0), -my / Scalar(2.0) );
      CScalar mc( mx / Scalar(2.0), my / Scalar(2.0) );

      tListX.push_back( T( u(i), m_F(i,j), mx ) );
      tListY.push_back( T( u(i), m_F(i,j), my ) );
      tListZ.push_back( TC( u(i), m_F(i,j), mz ) );
      tListC.push_back( TC( u(i), m_F(i,j), mc ) );

    }
  }

  // Complete sparse operator construction
  Eigen::SparseMatrix<Scalar> Dx( numF, numV );
  Eigen::SparseMatrix<Scalar> Dy( numF, numV );
  Eigen::SparseMatrix<CScalar> Dz( numF, numV );
  Eigen::SparseMatrix<CScalar> Dc( numF, numV );

  Dx.setFromTriplets( tListX.begin(), tListX.end() );
  Dy.setFromTriplets( tListY.begin(), tListY.end() );
  Dz.setFromTriplets( tListZ.begin(), tListZ.end() );
  Dc.setFromTriplets( tListC.begin(), tListC.end() );

  // Set member variables
  m_Dx = Dx;
  m_Dy = Dy;
  m_Dz = Dz;
  m_Dc = Dc;

};

///
/// Construct the mesh function averaging operators
///
template <typename Scalar, typename Index>
void DSEMpp::DSEM<Scalar, Index>::constructAveragingOperators() {

  typedef Eigen::Triplet<Scalar> T;

  // The number of vertices
  int numV = m_x.rows();

  // The number of faces
  int numF = m_F.rows();

  // A vector of face indices
  IndexVector fIDx(numF, 1);
  fIDx = IndexVector::LinSpaced(numF, 0, (numF-1));

  // Extract the internal angles of the 2D triangulation
  // NOTE: Some funky business with libigl's use of
  // Eigen::MatrixBase for templating requires temporary
  // dynamically sized storage for vertex/face connectivity lists
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> x = m_x;
  Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> F = m_F;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> fangles(numF, 3);
  igl::internal_angles( x, F, fangles );

  // Normalize the sums of internal angles around each vertex
  std::vector< std::vector<Scalar> > vangles;
  vangles.reserve( numV );

  std::vector< std::vector<Index> > vfIDx;
  vfIDx.reserve( numV );

  Vector vangleSum = Vector::Zero( numV, 1 );

  for( int i = 0; i < numV; i++ ) {

    std::vector<Scalar> tmpScalarVec;
    vangles.push_back( tmpScalarVec );

    std::vector<Index> tmpIndexVec;
    vfIDx.push_back( tmpIndexVec );

  }

  for( int i = 0; i < numF; i++ ) {
    for( int j = 0; j < 3; j++ ) {
      
      vangles[ m_F(i,j) ].push_back( fangles(i,j) );
      vfIDx[ m_F(i,j) ].push_back( (Index) i );
      vangleSum( m_F(i,j) ) += fangles(i,j);

    }
  }

  for( int i = 0; i < numV; i++ ) {
    for( int j = 0; j < vangles[i].size(); j++ ) {
      vangles[i][j] = vangles[i][j] / vangleSum(i);
    }
  }

  // Construct the vertex-to-face averaging operator
  std::vector<T> tListV2F;
  tListV2F.reserve( 3 * numF );

  for( int i = 0; i < numF; i++ ) {
    for( int j = 0; j < 3; j++ ) {
      tListV2F.push_back( T( fIDx(i), m_F(i,j), Scalar(1.0/3.0) ) );
    }
  }

  Eigen::SparseMatrix<Scalar> V2F( numF, numV );
  V2F.setFromTriplets( tListV2F.begin(), tListV2F.end() );

  m_V2F = V2F;

  // Construct the face-to-vertex averaging operator
  std::vector<T> tListF2V;
  tListF2V.reserve( 6 * numV ); // A rough estimate

  for( int i = 0; i < numV; i++ ) {
    for( int j = 0; j < vangles[i].size(); j++ ) {
      tListF2V.push_back( T( (Index) i, vfIDx[i][j], vangles[i][j] ) );
    }
  }

  Eigen::SparseMatrix<Scalar> F2V( numV, numF );
  F2V.setFromTriplets( tListF2V.begin(), tListF2V.end() );

  m_F2V = F2V;

};

// ======================================================================================
// MAPPING KERNEL FUNCTIONS
// ======================================================================================

///
/// Construct the components of the kernel that is integrated to find the
/// variation in the quasiconformal mapping under the variation of the
/// Beltrami coefficient
///
template <typename Scalar, typename Index>
DSEM_INLINE void DSEMpp::DSEM<Scalar, Index>::calculateMappingKernel (
    const CplxVector &w, Array &G1, Array &G2, Array &G3, Array &G4 ) const {

  // Number of vertices
  int numV = m_V.rows();

  // The squared partial derivative dw/dz defined on vertices
  CplxVector DWz2(numV, 1);
  DWz2.noalias() = m_F2V * m_Dz * w;
  DWz2 = (DWz2.array() * DWz2.array()).matrix();

  // Cached variables for conjugates
  CplxVector wC = w.conjugate();
  CplxVector DWz2C = DWz2.conjugate();

  // More cached variables to avoid duplicate computations
  CplxVector xi = w.array() * (Scalar(1.0) - w.array());
  CplxVector xiC = xi.conjugate();

  CplxVector pixi = Scalar(M_PI) * xi;
  CplxVector pixiC = pixi.conjugate();

  // Construct Complex Coefficients -----------------------------------------------------
  
  #pragma omp parallel for collapse(2)
  for( int i = 0; i < numV; i++ ) {
    for( int j = 0; j < numV; j++ ) {

      CScalar Anum = xi(i) * DWz2(j);
      CScalar Adenom = pixi(j) * (w(i) - w(j));

      if ( std::abs(Adenom) < Scalar(1e-14) ) {
        Adenom = CScalar(Scalar(0.0), Scalar(0.0));
      }

      CScalar Bnum = xi(i) * DWz2C(j);
      CScalar Bdenom = pixiC(j) * (Scalar(1.0) - wC(j) * w(i));

      if ( std::abs(Bdenom) < Scalar(1e-14) ) {
        Bdenom = CScalar(Scalar(0.0), Scalar(0.0));
      }

      CScalar A = Anum / Adenom; // The complex coefficient of nu
      CScalar B = Bnum / Bdenom; // The complex coefficient of nuC

      // Construct real mapping kernel coefficients
      G1(i,j) = A.real() + B.real();
      G2(i,j) = B.imag() - A.imag();
      G3(i,j) = A.imag() + B.imag();
      G4(i,j) = A.real() - B.real();

      // Deal with singularities
      // NOTE: 'isinfinite' is not a templated function
      // One workaround is to convert all input arguments to doubles
      if ( !std::isfinite( (double) G1(i,j)) ) { G1(i,j) = Scalar(0.0); }
      if ( !std::isfinite( (double) G2(i,j)) ) { G2(i,j) = Scalar(0.0); }
      if ( !std::isfinite( (double) G3(i,j)) ) { G3(i,j) = Scalar(0.0); }
      if ( !std::isfinite( (double) G4(i,j)) ) { G4(i,j) = Scalar(0.0); }

    }
  }

};

///
/// Calculates the change in the quasi conformal mapping for a given variation
/// in its associate Beltrami coefficient according to the Beltrami
/// Holomorphic Flow
///
template <typename Scalar, typename Index>
DSEM_INLINE void DSEMpp::DSEM<Scalar, Index>::calculateMappingUpdate (
    const CplxVector &nu, const Array &G1, const Array &G2,
    const Array &G3, const Array &G4, CplxVector &dW ) const {

  // Number of vertices
  int numV = m_V.rows();

  // Cannot transpose in place with Eigen
  CplxRowVector nuT = nu.transpose();

  // The real part of the variation
  RowVector nu1 = nuT.real();
  Array nu1Arr = nu1.replicate( numV, 1 );

  // The imaginary part of the variation
  RowVector nu2 = nuT.imag();
  Array nu2Arr = nu2.replicate( numV, 1 );

  // Calculate the update step
  CplxArray dWArr(numV, numV);
  dWArr.real() = G1 * nu1Arr + G2 * nu2Arr;
  dWArr.imag() = G3 * nu1Arr + G4 * nu2Arr;
  dWArr = dWArr * m_vertexAreaMat;

  dW = dWArr.rowwise().sum();

};

// ======================================================================================
// ENERGY AND ENERGY GRADIENT FUNCTIONS
// ======================================================================================

///
/// Construct the sparse operator that calculates global gradients
/// from vertex based gradients
///
template <typename Scalar, typename Index>
DSEM_INLINE void DSEMpp::DSEM<Scalar, Index>::energyGradientOperator (
    const Matrix &DXu, const Matrix &DXv,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &eVec,
    const Vector &l, const Vector &L, const Vector &tarA_E,
    RowVector &GOp ) const {

  // Number of vertices
  int numV = m_V.rows();

  // Number of edges
  int numE = m_E.rows();

  // Each vertex based gradient is a matrix of size (2 x N) where N is
  // the number of real unknown living at each vertex with respect to
  // the given gradient calculation
  
  typedef Eigen::Triplet<Scalar> T;
  
  // Create surface derivative chain rule operator --------------------------------------
  
  std::vector<T> tList1;
  tList1.reserve(6 * numV);

  for( int i = 0; i < numV; i++ ) {
    for( int j = 0; j < 3; j++ ) {

      tList1.push_back( T(i+(j*numV), i, DXu(i,j)) );
      tList1.push_back( T(i+(j*numV), i+numV, DXv(i,j)) );

    }
  }

  Eigen::SparseMatrix<Scalar> GOp1( 3*numV, 2*numV );
  GOp1.setFromTriplets( tList1.begin(), tList1.end() );

  // Create vertex-to-edge operator -----------------------------------------------------
  
  std::vector<T> tList2;
  tList2.reserve(6 *numE);

  for( int i = 0; i < numE; i++ ) {
    for( int j = 0; j < 3; j++ ) {

      tList2.push_back( T(i+(j*numE), (int) m_E(i,1)+(j*numV), Scalar(1.0)) );
      tList2.push_back( T(i+(j*numE), (int) m_E(i,0)+(j*numV), -Scalar(1.0)) );

    }
  }

  Eigen::SparseMatrix<Scalar> GOp2( 3*numE, 3*numV );
  GOp2.setFromTriplets( tList2.begin(), tList2.end() );

  // Create edge-sum operator -----------------------------------------------------------
  
  // UNSCALED OPERATOR
  // ArrayVec WEV = Scalar(2.0) * tarA_E.array() * (l.array() - L.array()) / l.array();

  // SCALED OPERATOR
  ArrayVec WEV = Scalar(2.0) * tarA_E.array() * (l.array() - L.array())
    / ( tarA_E.sum() * l.array() * L.array() * L.array() );

  Vector GOp3T( 3*numE, 1 );
  GOp3T << (WEV * eVec.col(0).array()).matrix(),
       (WEV * eVec.col(1).array()).matrix(),
       (WEV * eVec.col(2).array()).matrix();

  RowVector GOp3 = GOp3T.transpose();

  // Combine to create the complete operator --------------------------------------------
  
  GOp = GOp3 * GOp2 * GOp1;

};
  
///
/// Calculate the DSEM energy functional for a given configuration
///
template <typename Scalar, typename Index>
Scalar DSEMpp::DSEM<Scalar, Index>::calculateEnergy (
    const Vector &L, const Vector &tarA_E, const Vector &tarA_V,
    const CplxVector &mu, const CplxVector &w,
    const Scalar phi, const CScalar &w0,
    Vector &l ) const {

  // Number of vertices
  int numV = m_V.rows();

  // ------------------------------------------------------------------------------------
  // Calculate DSEM Energy
  // ------------------------------------------------------------------------------------
  
  // Apply Mobius mapping to quasiconformal mapping
  CplxVector uC = std::exp( CScalar(Scalar(0.0), phi) ) *
    ( (w.array() - w0) / (Scalar(1.0) - std::conj(w0) * w.array()) ).matrix();

  // Calculate 3D vertex positions
  Matrix X(numV, 3);
  m_NNI( uC.real(), uC.imag(), X );

  // Calculate 3D edge lengths
  igl::edge_lengths( X, m_E, l );

  // UNSCALED ENERGY
  // Scalar E = ( tarA_E.array() * (l.array() - L.array()) * (l.array() - L.array()) ).sum();
  
  // SCALED ENERGY
  Scalar E = ( tarA_E.array() * (l.array() - L.array()) * (l.array() - L.array())
      / (L.array() * L.array()) ).sum() / tarA_E.sum();

  // ------------------------------------------------------------------------------------
  // Calculate Conformal Deviation Energy
  // ------------------------------------------------------------------------------------
  
  if ( m_param.CC > Scalar(0.0) ) {

    Scalar ECC = ( mu.array() * mu.array().conjugate() * tarA_V.array() ).sum().real();
    E += m_param.CC * ECC;

  }

  // ------------------------------------------------------------------------------------
  // Calculate Quasiconformal Smoothness Energy
  // ------------------------------------------------------------------------------------
  
  if ( m_param.SC > Scalar(0.0) ) {

    // TODO
    
  }

  // ------------------------------------------------------------------------------------
  // Calculate Bound Constraint Energy on the Beltrami Coefficient
  // ------------------------------------------------------------------------------------
  
  if ( m_param.DC > Scalar(0.0) ) {

    Scalar EDC = ( Scalar(1.0) - mu.array().abs() ).log().sum();
    E -= EDC / m_param.DC;

  }

  return E;

};

///
/// Calculate the DSEM energy functional and gradients for a given configuration
///
template <typename Scalar, typename Index>
Scalar DSEMpp::DSEM<Scalar, Index>::calculateEnergyAndGrad (
    const Vector &L, const Vector &tarA_E, const Vector &tarA_V,
    const CplxVector &mu, const CplxVector &w,
    const Scalar phi, const CScalar &w0,
    const Array &G1, const Array &G2, const Array &G3, const Array &G4,
    CplxVector &gradMu, CScalar &gradW0,
    Scalar &gradPhi, Scalar &gradUNorm, Vector &l ) const {

  std::cout << "Check EG 0" << std::endl;

  // Number of vertices
  int numV = m_V.rows();

  // Number of edges
  int numE = m_E.rows();

  // ------------------------------------------------------------------------------------
  // Calculate DSEM Energy
  // ------------------------------------------------------------------------------------
  
  // Apply Mobius mapping to quasiconformal mapping
  CplxVector uC = std::exp( CScalar(Scalar(0.0), phi) ) *
    ( (w.array() - w0) / (Scalar(1.0) - std::conj(w0) * w.array()) );

  std::cout << "Check EG 1" << std::endl;

  // The real and imaginary parts of the updated mapping
  Vector u = uC.real();
  Vector v = uC.imag();

  std::cout << "U size = " << u.size() << std::endl;
  std::cout << "V size = " << v.size() << std::endl;

  // Calculate 3D vertex positions
  Matrix X(numV, 3);
  Matrix DXu(numV, 3);
  Matrix DXv(numV, 3);
  m_NNI( u, v, X, DXu, DXv );

  std::cout << "Check EG 2" << std::endl;

  // Calculate directe 3D edge vectors
  Eigen::Matrix<Scalar, Eigen::Dynamic, 3> eVec(numE, 3);
  for (int i = 0; i < numE; i++) {
    eVec.row(i) = X.row(m_E(i,1)) - X.row(m_E(i,0));
  }

  // Calculate 3D edge lengths
  l = (eVec.array() * eVec.array()).rowwise().sum().sqrt();

  // UNSCALED ENERGY
  // Scalar E = ( tarA_E.array() * (l.array() - L.array()) * (l.array() - L.array()) ).sum();
  
  // SCALED ENERGY
  Scalar E = ( tarA_E.array() * (l.array() - L.array()) * (l.array() - L.array())
      / (L.array() * L.array()) ).sum() / tarA_E.sum();

  std::cout << "Check EG 3" << std::endl;

  // ------------------------------------------------------------------------------------
  // Calculate DSEM Energy Gradients
  // ------------------------------------------------------------------------------------
  
  // Calculate energy gradient operator
  RowVector GOp(1, 2*numV);
  this->energyGradientOperator( DXu, DXv, eVec, l, L, tarA_E, GOp );

  std::cout << "Check EG 4" << std::endl;

  // Calculate the norm of the gradient with respect to the fully composed parameterization
  gradUNorm = ( GOp * m_speye ).norm();

  // Construct the 'phi' gradient -------------------------------------------------------
  Vector dudphi( 2*numV, 1 );
  dudphi << -v, u;

  gradPhi = GOp * dudphi;

  // Construct the 'w0' gradient --------------------------------------------------------
  Scalar b = w0.real();
  Scalar c = w0.imag();

  CplxArrayVec denom = (Scalar(1.0) - std::conj(w0) * w.array());
  denom = denom * denom;

  CplxArrayVec dUdb = std::exp( CScalar(Scalar(0.0), phi) ) *
    ( (w.array() * w.array() -
       CScalar(Scalar(0.0), Scalar(2.0) * c) * w.array() - Scalar(1.0)) / denom );

  CplxArrayVec dUdc = -CScalar(Scalar(0.0), Scalar(1.0)) *
    std::exp( CScalar(Scalar(0.0), phi) ) *
    ( (w.array() * w.array() - Scalar(2.0) * b * w.array() + Scalar(1.0)) / denom );

  Matrix dUdw0( 2*numV, 2 );
  dUdw0 << dUdb.real(), dUdc.real(),
        dUdb.imag(), dUdc.imag();

  Eigen::Matrix<Scalar, 1, 2> gradW0R = GOp * dUdw0;
  gradW0 = CScalar( gradW0R(0), gradW0R(1) );

  std::cout << "Check EG 5" << std::endl;

  // Construct the 'mu' gradient --------------------------------------------------------
  CplxArrayVec dUdw1 = std::exp( CScalar(Scalar(0.0), phi) ) *
    (Scalar(1.0) - w0 * std::conj(w0)) / denom;

  #pragma omp parallel for if (numV > 500)
  for( int i = 0; i < numV; i++ ) {

    /* I'm including this bit of inefficient code as a comment
     * to make the structure of the chain rule derivatives for mu
     * more transparent

    ArrayVec du1dw1 = dUdw1.real();
    ArrayVec du2dw1 = dUdw1.imag();
    ArrayVec du1dw2 = -du2dw1;
    ArrayVec du2dw2 = du1dw1;

    ArrayVec dw1dmu1 = G1.col(i);
    ArrayVec dw1dmu2 = G2.col(i);
    ArrayVec dw2dmu1 = G3.col(i);
    ArrayVec dw2dmu2 = G4.col(i);

    ArrayVec du1dmu1 = du1dw1 * dw1dmu1 + du1dw2 * dw2dmu1;
    ArrayVec du2dmu1 = du2dw1 * dw1dmu1 + du2dw2 * dw2dmu1;
    ArrayVec du1dmu2 = du1dw1 * dw1dmu2 + du1dw2 * dw2dmu2;
    ArrayVec du2dmu2 = du2dw1 * dw1dmu2 + du2dw2 * dw2dmu2;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 2> dUdmu( 2*numV, 2 );
    dUdmu << du1dmu1, du1dmu2,
          du2dmu1, du2dmu2;

    */

    Eigen::Matrix<Scalar, Eigen::Dynamic, 2> dUdmu( 2*numV, 2 );
    dUdmu << dUdw1.real() * G1.col(i) - dUdw1.imag() * G3.col(i),
             dUdw1.real() * G2.col(i) - dUdw1.imag() * G4.col(i),
             dUdw1.imag() * G1.col(i) + dUdw1.real() * G3.col(i),
             dUdw1.imag() * G2.col(i) + dUdw1.real() * G4.col(i);

    Eigen::Matrix<Scalar, 1, 2> gradMuR = GOp * dUdmu;
    gradMu(i) = CScalar( gradMuR(0), gradMuR(1) );

  }

  std::cout << "Check EG 6" << std::endl;

  // ------------------------------------------------------------------------------------
  // Calculate Conformal Deviation Energy and Gradient
  // ------------------------------------------------------------------------------------
  
  if ( m_param.CC > Scalar(0.0) ) {

    Scalar ECC = ( mu.array() * mu.array().conjugate() * tarA_V.array() ).sum().real();
    E += m_param.CC * ECC;

    gradMu = ( gradMu.array() +
        Scalar(2.0) * m_param.CC * mu.array() * tarA_V.array() ).matrix();

  }

  // ------------------------------------------------------------------------------------
  // Calculate Quasiconformal Smoothness Energy
  // ------------------------------------------------------------------------------------
  
  if ( m_param.SC > Scalar(0.0) ) {

    // TODO
    
  }

  // ------------------------------------------------------------------------------------
  // Calculate Bound Constraint Energy on the Beltrami Coefficient
  // ------------------------------------------------------------------------------------
  
  if ( m_param.DC > Scalar(0.0) ) {

    Scalar EDC = ( Scalar(1.0) - mu.array().abs() ).log().sum();
    E -= EDC / m_param.DC;

    ArrayVec absMu = mu.array().abs();
    gradMu = ( gradMu.array() +
        mu.array() / ( absMu * (Scalar(1.0) - absMu) ) / m_param.DC ).matrix();

  }

  return E;

};

// ======================================================================================
// BELTRAMI HOLOMORPHIC FLOW
// ======================================================================================

///
/// Assemble the various quantity lists into a single global vector
///
template <typename Scalar, typename Index>
DSEM_INLINE void DSEMpp::DSEM<Scalar, Index>::assembleGlobalQuantities (
    const CplxVector &muQ, const CScalar &w0Q, const Scalar &phiQ, Vector &xQ ) const {

  xQ << phiQ, w0Q.real(), w0Q.imag(), muQ.real(), muQ.imag();

};

///
/// Disassemble a global vector into the various quantitty lists
///
template <typename Scalar, typename Index>
DSEM_INLINE void DSEMpp::DSEM<Scalar, Index>::disassembleGlobalQuantities (
    const Vector &xQ, CplxVector &muQ, CScalar &w0Q, Scalar &phiQ ) const {

  phiQ = xQ(0);

  w0Q = CScalar( xQ(1), xQ(2) );

  int numV = muQ.size();
  for( int i = 0; i < numV; i++ ) {

    muQ(i) = CScalar( xQ(i+3), xQ(i+3+numV) );

  }

};

///
/// Calculate the SEM distance for a given discrete metric
///
template <typename Scalar, typename Index>
Scalar DSEMpp::DSEM<Scalar, Index>::operator() (
    const Vector &L,
    const CplxVector &initMu, const CplxVector &initMap,
    const Scalar &initPhi, const CScalar &initW0,
    CplxVector &mu, CplxVector &w,
    Scalar &phi, CScalar &w0, Vector &l ) const {

  // Number of vertices
  int numV = m_V.rows();

  // Number of faces
  int numF = m_F.rows();

  // Number of edges
  int numE = m_E.rows();

  // ------------------------------------------------------------------------------------
  // Input Processing
  // ------------------------------------------------------------------------------------
  
  // Process target metric --------------------------------------------------------------
  
  // Check target edge list size
  if (L.size() != numE) {
    throw std::runtime_error("Improperly sized target edge list");
  }

  // Check target edge list entries
  for( int i = 0; i < numE; i++ ) {
    if ( (!std::isfinite(L(i))) || (L(i) <= 0) ) {
      throw std::runtime_error("Invalid target edge list");
    }
  }

  // Associate target edges to faces
  Eigen::Array<Scalar, Eigen::Dynamic, 3> L_F( numF, 3 );
  for( int i = 0; i < numF; i++ ) {
    for( int j = 0; j < 3; j++ ) {
      L_F(i,j) = L( m_FE(i,j) );
    }
  }

  // Check that the input edge list satisfies the triangle inequality
  ArrayVec S_F = L_F.rowwise().sum(); // 2x the semi-perimeter in Heron's formula

  Eigen::Array<bool, Eigen::Dynamic, 3> badMetric(numF, 3);
  badMetric = (S_F.replicate(1,3) - Scalar(2.0) * L_F.array()) <= Scalar(0.0);
  if ( badMetric.any() ) {
    throw std::runtime_error("Target metric does NOT satisfy the triangle inequality");
  }

  // Calculate target face areas
  S_F = S_F / Scalar(2.0);
  
  Vector tarA_F = ( ( S_F * (S_F - L_F.col(0)) *
        (S_F - L_F.col(1)) * (S_F - L_F.col(2)) ).sqrt() ).matrix();

  // Calculate target edge areas
  Vector tarA_E = Vector::Zero( numE, 1 );
  for( int i = 0; i < numE; i++ ) {
    for( int j = 0; j < m_EF[i].size(); j++ ) {

      tarA_E(i) += tarA_F( m_EF[i][j] );

    }
  }

  tarA_E = (tarA_E.array() / Scalar(3.0)).matrix();

  // Calculate target vertex areas
  Vector tarA_V = Vector::Zero( numV, 1 );
  for( int i = 0; i < numF; i++ ) {
    for( int j = 0; j < 3; j++ ) {

      tarA_V( m_F(i,j) ) += tarA_F(i);

    }
  }

  tarA_V = (tarA_V.array() / Scalar(3.0)).matrix();

  // Initial guess processing -----------------------------------------------------------
  // TODO
  
  // Set optimization parameters according to the initial guess
  mu = initMu;
  w = initMap;
  phi = initPhi;
  w0 = initW0;

  // Clip the boundary vertices of w to the unit circle
  DSEMpp::clipToUnitCircle( m_bdyIDx, w );
  
  // ------------------------------------------------------------------------------------
  // Run the Beltrami Holomorphic Flow via L-BFGS
  // ------------------------------------------------------------------------------------
  
  // Optimization Pre-Processing --------------------------------------------------------
  
  // The objective function value
  Scalar fx;
  
  // The number of unknowns to determine
  const int numUnknowns = 3 + 2 * numV;
  
  // Approximation to the Hessian matrix
  DSEMpp::BFGSMat<Scalar> bfgs;
  bfgs.reset(numUnknowns, m_param.m);

  // Current global unknown list
  Vector x(numUnknowns, 1);

  // Old global unknown list
  Vector xp(numUnknowns, 1);

  // Old quasiconformal map
  CplxVector wp(numV, 1);

  // New global gradient
  Vector grad(numUnknowns, 1);
  
  // Old global gradient
  Vector gradp(numUnknowns, 1);

  // The update direction
  Vector drt(numUnknowns, 1);

  // The update direction for the quasiconformal mapping
  CplxVector dw(numV, 1);

  // The length of the lag for objective function values to test convergence
  const int fpast = m_param.past;
  
  // History of objective function values
  Vector fxp;
  if ( m_param.past > 0 ) { fxp.resize(fpast); }

  // Storage for calculating the Beltrami coefficient from the mapping
  CplxVector muF( numF, 1 );

  // Handle the '0th' Iteration ---------------------------------------------------------
  
  // Calculate the mapping kernel
  Array G1( numV, numV );
  Array G2( numV, numV );
  Array G3( numV, numV );
  Array G4( numV, numV );
  calculateMappingKernel( w, G1, G2, G3, G4 );

  // Evaluate the function and gradient for initial configuration
  CplxVector gradMu( numV, 1 );
  CScalar gradW0( Scalar(0.0), Scalar(0.0) );
  Scalar gradPhi = Scalar(0.0);
  Scalar gradUNorm = Scalar(0.0);

  fx = calculateEnergyAndGrad( L, tarA_E, tarA_V,
      mu, w, phi, w0, G1, G2, G3, G4,
      gradMu, gradW0, gradPhi, gradUNorm, l );

  // Assemble the global unknown vector
  assembleGlobalQuantities( mu, w0, phi, x );

  // Assemble the global gradient vector
  assembleGlobalQuantities( gradMu, gradW0, gradPhi, grad );

  // Updated vector norms
  Scalar xnorm = x.norm();
  Scalar gnorm = grad.norm();

  if (true) {

    std::cout << "(0)"
      << " ||du|| = " << std::fixed << std::setw(10) << std::setprecision(7) << gradUNorm
      << " ||dx|| = " << std::setw(10) << std::setprecision(7) << gnorm
      << " ||x|| = " << std::setprecision(4) << std::setw(6) << xnorm
      << " f(x) = " << std::setw(15) << std::setprecision(10) << fx << std::endl;

  }

  if ( fpast > 0 ) { fxp[0] = fx; }

  // Early exit if the initial guess is already a minimizer
  if ( gnorm <= m_param.epsilon || gnorm <= m_param.epsilonRel * xnorm ) {
    return fx;
  }

  // Initial update direction
  drt.noalias() = -grad;

  // Disassemble the update dirrection
  CplxVector dmu(numV, 1);
  CScalar dw0( Scalar(0.0), Scalar(0.0) );
  Scalar dphi = Scalar(0.0);

  disassembleGlobalQuantities( drt, dmu, dw0, dphi );

  // Calculate the inital update direction for the quasiconformal mapping
  calculateMappingUpdate(dmu, G1, G2, G3, G4, dw);

  // Initial step size
  Scalar step = Scalar(1.0) / drt.norm();

  // Handle All Subsequent Iterations ---------------------------------------------------
  
  int iterNum = 1;
  for( ; ; ) {

    xp.noalias() = x; // Save the current global unknown vector
    wp.noalias() = w; // Save the current quasiconformal mapping
    gradp.noalias() = grad; // Save the current global gradient vector

    // Perform line search to update unknowns
    DSEMpp::LineSearchBacktracking<Scalar, Index>::LineSearch(
        *this, m_param, L, tarA_E, tarA_V, xp, wp, drt, dw,
        fx, x, w, grad, gradUNorm, step, l );

    // Disassemble the global unknown vector
    disassembleGlobalQuantities( x, mu, w0, phi );

    // New vector norms
    xnorm = x.norm(); 
    gnorm = grad.norm();

    if (true) {

      std::cout << "(" << std::setw(2) << iterNum << ")"
        << " ||du|| = " << std::fixed << std::setw(10)
        << std::setprecision(7) << gradUNorm
        << " ||dx|| = " << std::setprecision(7) << std::setw(10) << gnorm
        << " ||x|| = " << std::setprecision(4) << std::setw(6) << xnorm
        << " f(x) = " << std::setw(15) << std::setprecision(14)
        << fx << std::endl;

    }

    // CONVERGENCE TEST -- Gradient
    if ( gnorm <= m_param.epsilon || gnorm <= m_param.epsilonRel * xnorm ) {
      return fx;
    }

    // CONVERGENCE TEST -- Objective function value
    if ( fpast > 0 ) {

      const Scalar fxd = fxp[iterNum % fpast];
      Scalar smallChange = m_param.delta * std::max(std::max(abs(fx), abs(fxd)), Scalar(1.0));

      bool longEnough = iterNum >= fpast;
      bool slowChange = std::abs(fxd-fx) <= smallChange;

      if( longEnough && slowChange ) {
        return fx;
      }

      fxp[iterNum % fpast] = fx;

    }

    // CONVERGENCE TEST -- Maximum number of iterations
    if ( m_param.maxIterations != 0 && iterNum >= m_param.maxIterations ) {
      return fx;
    }

    // Update s and y
    // s_{k+1} = x_{k+1} - x_k
    // y_{k+1} = g_{k+1} - g_k
    bfgs.addCorrection( x - xp, grad - gradp );

    // Recursive formula to compute d = -H * g
    bfgs.applyHv( grad, -Scalar(1.0), drt );

    // Disassemble the update direction
    disassembleGlobalQuantities(drt, dmu, dw0, dphi);

    // Calculate the update direction for the quasiconformal mapping
    calculateMappingUpdate( dmu, G1, G2, G3, G4, dw );

    // Reset step = 1.0 as initial guess for the next line search
    step = Scalar(1.0);

    // Increment the iteration count
    iterNum++;

  }

  return fx;

};

//TODO: Add explicit template instantiation
#ifdef DSEM_STATIC_LIBRARY
#endif
