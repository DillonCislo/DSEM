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

#ifndef _DSEM_H_
#define _DSEM_H_

#include "dsemInline.h"

#include <vector>
#include <complex>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "DSEMParam.h"
#include "../external/NNIpp/include/NaturalNeighborInterpolant/NaturalNeighborInterpolant.h"
#include "../external/NNIpp/include/NaturalNeighborInterpolant/NNIParam.h"

namespace DSEMpp {

  ///
  /// A class to calculate the Shape Equivalent Metric (SEM) distance for a given
  /// discrete metric relative to a particular surface. A metric is considered
  /// to be an element of the space of SEMs for a particular surface if that metric
  /// can be found as the first fundamental form of a surface with an arbitrary
  /// parameterization.
  ///
  /// Templates:
  ///
  ///   Scalar    The input type of data points and function values
  ///   Index     The data type of the triangulation indices
  ///
  template <typename Scalar, typename Index>
  class DSEM {

    public:

      typedef std::complex<Scalar> CScalar;
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
      typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic> RowVector;
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
      typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> ArrayVec;
      typedef Eigen::Array<Scalar, 1, Eigen::Dynamic> ArrayRowVec;
      typedef Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> Array;
      typedef Eigen::Matrix<CScalar, Eigen::Dynamic, 1> CplxVector;
      typedef Eigen::Matrix<CScalar, 1, Eigen::Dynamic> CplxRowVector;
      typedef Eigen::Matrix<CScalar, Eigen::Dynamic, Eigen::Dynamic> CplxMatrix;
      typedef Eigen::Array<CScalar, Eigen::Dynamic, 1> CplxArrayVec;
      typedef Eigen::Array<CScalar, Eigen::Dynamic, Eigen::Dynamic> CplxArray;
      typedef Eigen::Matrix<Index, Eigen::Dynamic, 1> IndexVector;
      typedef Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> IndexMatrix;
      typedef Eigen::Array<Index, Eigen::Dynamic, 1> IndexArrayVec;
      typedef Eigen::Array<Index, Eigen::Dynamic, Eigen::Dynamic> IndexArray;

      // Mesh Properties ----------------------------------------------------------------

      // #F by 3 matrix. Discrete surface face connectivity list
      Eigen::Matrix<Index, Eigen::Dynamic, 3> m_F;

      // #V by 3 matrix. 3D discrete surface vertex coordinate list
      Eigen::Matrix<Scalar, Eigen::Dynamic, 3> m_V;

      // #V by 2 matrix. 2D pullback coordinate list.
      // Defines the domain of parameterization
      Eigen::Matrix<Scalar, Eigen::Dynamic, 2> m_x;

      // #E by 2 matrix. Discrete surface edge connectivity list
      Eigen::Matrix<Index, Eigen::Dynamic, 2> m_E;

      // #E by 3 matrix. m_EF(i,j) is the ID of the edge opposite
      // vertex j in face i
      Eigen::Matrix<Index, Eigen::Dynamic, 3> m_FE;

      // #E by 1 vector of vectors. Contains the IDs of faces attached
      // to unoriented edges
      std::vector<std::vector<Index> > m_EF;

      // #V by #F sparse matrix operator. Averages face-based quantities onto
      // vertices using angle weighting in the domain of parameterization
      Eigen::SparseMatrix<Scalar> m_F2V;

      // #F by #V sparse matrix operator. Averages vertex-based quantities
      // onto faces
      Eigen::SparseMatrix<Scalar> m_V2F;

      // #V by 1 list of barycentric vertex areas in the 2D pullback mesh
      Vector m_vertexAreas;

      // #V by #V array. Holds a replicated version of 'm_vertexAreas' used
      // to calculate the update in the quasiconformal mapping
      // Equal to m_vertexAreas.transpose().replicate(#V, 1)
      Array m_vertexAreaMat;

      // A vector of all vertex IDs on the mesh boundary
      IndexVector m_bdyIDx;

      // A vector of all vertex IDs in the bulk
      IndexVector m_bulkIDx;

      // #2V by #2V sparse identity matrix. Used to calculate abbreviated gradients
      // with respect to fully composed parameterization
      Eigen::SparseMatrix<Scalar> m_speye;

      // Mesh Differential Operators ----------------------------------------------------

      // #F by #V sparse matrix operator df/dx
      Eigen::SparseMatrix<Scalar> m_Dx;

      // #F by #V sparse matrix operator df/dy
      Eigen::SparseMatrix<Scalar> m_Dy;

      // #F by #V sparse matrix operator df/dz
      Eigen::SparseMatrix<CScalar> m_Dz;

      // #F by #V sparse matrix operator df/dz*
      Eigen::SparseMatrix<CScalar> m_Dc;

      // Surface Interpolant Properties -------------------------------------------------

      // The natural neighbor interpolant representing the discrete surface
      NNIpp::NaturalNeighborInterpolant<Scalar> m_NNI;

      // Optimization Properties --------------------------------------------------------

      DSEMpp::DSEMParam<Scalar> m_param;

    public:

      ///
      /// Default constructor
      ///
      /// Inputs:
      ///
      ///   F           #F by 3 face connectivity list
      ///   V           #V by 3 3D vertex coordinate list
      ///   x           #V by 2 2D vertex coordinate list
      ///   dParam      A 'DSEMParam' class containing the optimization parameters
      ///   nniParam    An 'NNIParam' class containing the parameters needed to construct
      ///               the surface interpolant
      ///
      DSEM( const Eigen::Matrix<Index, Eigen::Dynamic, 3> &F,
          const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &V,
          const Eigen::Matrix<Scalar, Eigen::Dynamic, 2> &x,
          const DSEMParam<Scalar> &dsemParam,
          const NNIpp::NNIParam<Scalar> &nniParam );

      ///
      /// Calculate the SEM distance for a given discrete metric
      ///
      /// Inputs:
      ///
      ///   L         #E by 1 list of target edge lengths
      ///
      ///   initMu    #V by 1 initial guess for the Beltrami coefficient
      ///             that specifies the mapping corresponding to the minimum
      ///             distance SEM
      ///
      ///   initMap   #V by 1 complex representation of the quasiconformal
      ///             mapping specified by 'initMu'
      ///
      ///   initPhi   Initial guess for the rotation angle of the Mobius
      ///             transformation corresponding to the minimum distance SEM
      ///
      ///   initW0    Initial guess for the complex parameter of the Mobius
      ///             transformation corresponding to the minimum distance SEM
      ///
      /// Outputs:
      ///
      ///   D     The SEM distance
      ///
      ///   mu    #V by 1 complex representation of the Beltrami coefficient
      ///         that specifies the mapping corresponding to the minimum distance SEM
      ///
      ///   w     #V by 1 complex representation of the quasiconformal mapping
      ///         specified by mu. NOTE: Does NOT include the post-composition with
      ///         a Mobius transformation
      ///
      ///   phi   Overall rotation part of the Mobius transformation corresponding
      ///         to the minimum distance SEM
      ///
      ///   w0    Complex paramter of the Mobius transformation corresponding to the
      ///         minimum distance SEM
      ///
      ///   l     #E by 1 list of edge lengths. This is the discrete SEM that
      ///         minimuzed Dsem for the target metric
      ///
      Scalar operator() ( const Vector &L,
          const CplxVector &initMu, const CplxVector &initMap,
          const Scalar &initPhi, const CScalar &initW0,
          CplxVector &mu, CplxVector &w,
          Scalar &phi, CScalar &w0, Vector &l ) const;


    public:

      ///
      /// Calculate the DSEM energy functional for a given configuration
      ///
      /// Inputs:
      ///
      ///   L       #E by 1 list of target edge lengths
      ///
      ///   tarA_E  #E by 1 list of target areas assocated to edges
      ///
      ///   tarA_V  #V by 1 list of target areas associated to vertices
      ///
      ///   mu      #V by 1 list of complex Beltrami coefficients specifying
      ///           the mapping corresponding to the minimum distance SEM
      ///
      ///   w       #V by 1 complex representation of the quasiconformal mapping
      ///           corresponding to the minimum distance SEM. NOTE: does NOT
      ///           include the post-composition with a Mobius transformation
      ///
      ///   phi     Overall rotation part of the Mobius transformation that
      ///           specifies the mapping corresponding to the minimum distance SEM
      ///
      ///   w0      Complex parameter needed to construct the Mobius transformation
      ///           that specifies the mapping corresponding to the minimum distance SEM
      ///
      /// Outputs:
      ///
      ///   E       The total energy for the input configuration
      ///
      ///   l       #E by 1 list of calculated edge lengths
      ///
      Scalar calculateEnergy (
          const Vector &L, const Vector &tarA_E, const Vector &tarA_V,
          const CplxVector &mu, const CplxVector &w,
          const Scalar phi, const CScalar &w0,
          Vector &l ) const;

      ///
      /// Calculate the DSEM energy functional and gradients for a given configuration
      ///
      /// Inputs:
      ///
      ///   L       #E by 1 list of target edge lengths
      ///
      ///   tarA_E  #E by 1 list of target areas assocated to edges
      ///
      ///   tarA_V  #V by 1 list of target areas associated to vertices
      ///
      ///   mu      #V by 1 list of complex Beltrami coefficients specifying
      ///           the mapping corresponding to the minimum distance SEM
      ///
      ///   w       #V by 1 complex representation of the quasiconformal mapping
      ///           corresponding to the minimum distance SEM. NOTE: does NOT
      ///           include the post-composition with a Mobius transformation
      ///
      ///   phi     Overall rotation part of the Mobius transformation that
      ///           specifies the mapping corresponding to the minimum distance SEM
      ///
      ///   w0      Complex parameter needed to construct the Mobius transformation
      ///           that specifies the mapping corresponding to the minimum distance SEM
      ///
      ///   G1      #V by #V array. Coefficient of nu1 in the real part of K
      ///
      ///   G2      #V by #V array. Coefficient of nu2 in the real part of K
      ///
      ///   G3      #V by #V array. Coefficient of nu1 in the imaginary part of K
      ///
      ///   G4      #V by #V array. Coefficient of nu2 in the imaginary part of K
      ///
      /// Outputs:
      ///
      ///   E           The total energy for the input configuration
      ///
      ///   gradMu      #V by 1 complex gradient vector wrt the Beltrami coefficient
      ///
      ///   gradW0      Scalar complex gradient wrt the Mobius parameter
      ///
      ///   gradPhi     Scalar gradient wrt the overal rotation
      ///
      ///   gradUNorm   Scalar norm of the gradient with respect to the fully
      ///               composed parameterization
      ///
      ///   l           #E by 1 list of calculated edge lengths
      ///
      Scalar calculateEnergyAndGrad (
          const Vector &L, const Vector &tarA_E, const Vector &tarA_V,
          const CplxVector &mu, const CplxVector &w,
          const Scalar phi, const CScalar &w0,
          const Array &G1, const Array &G2, const Array &G3, const Array &G4,
          CplxVector &gradMu, CScalar &gradW0,
          Scalar &gradPhi, Scalar &gradUNorm, Vector &l ) const;

      ///
      /// Construct the operator that calculates global gradients
      /// from vertex based gradients
      ///
      /// Inputs:
      ///
      ///   DXu     #V by 3 array. The gradient of the interpolated 3D vertex
      ///           positions with respect to the real part of the Mobius mapping
      ///
      ///   DXv     #V by 3 array. The gradient of the interpolated 3D vertex
      ///           positions with respect to the imaginary part of the Mobius mapping
      ///
      ///   eVec    #E by 2 array. The directed edge vectors of the interpolated
      ///           3D configuration
      ///
      ///   l       #E by 1 array. The edge lengths of the interpolated 3D configuration
      ///
      ///   L       #E by 1 array. The target edge lengths
      ///
      ///   tarA_E  #E by 1 array. The target edge areas
      ///
      /// Outputs:
      ///
      ///   GOp     1 by (2*#V) operator that compiles vertex based gradients
      ///           into complete energy gradients
      ///
      DSEM_INLINE void energyGradientOperator (
          const Matrix &DXu, const Matrix &DXv,
          const Eigen::Matrix<Scalar, Eigen::Dynamic, 3> &eVec,
          const Vector &l, const Vector &L, const Vector &tarA_E,
          RowVector &GOp ) const;

      ///
      /// Construct the components of the kernel that is integrated to find the
      /// variation in the quasiconformal mapping under the variation of the
      /// Beltrami coefficient
      ///
      /// Inputs:
      ///
      ///   w     #V by 1 complex representation of a quasiconformal mapping
      ///
      /// Outputs:
      ///
      ///   G1      #V by #V array. Coefficient of nu1 in the real part of K
      ///
      ///   G2      #V by #V array. Coefficient of nu2 in the real part of K
      ///
      ///   G3      #V by #V array. Coefficient of nu1 in the imaginary part of K
      ///
      ///   G4      #V by #V array. Coefficient of nu2 in the imaginary part of K
      ///
      DSEM_INLINE void calculateMappingKernel ( const CplxVector &w,
          Array &G1, Array &G2, Array &G3, Array &G4 ) const;

      ///
      /// Calculates the change in the quasiconformal mapping for a given variation
      /// in its associated Beltrami coefficient according to the Beltrami
      /// Holomorphic Flow
      ///
      /// Inputs:
      ///
      ///   nu      #V by 1 complex representation of the variation in the Beltrami
      ///           coefficient
      ///
      ///   G1      #V by #V array. Coefficient of nu1 in the real part of K
      ///
      ///   G2      #V by #V array. Coefficient of nu2 in the real part of K
      ///
      ///   G3      #V by #V array. Coefficient of nu1 in the imaginary part of K
      ///
      ///   G4      #V by #V array. Coefficient of nu2 in the imaginary part of K
      ///
      /// Outputs:
      ///
      ///   dW      #V by 1 change in the mapping
      ///
      DSEM_INLINE void calculateMappingUpdate ( const CplxVector &nu,
          const Array &G1, const Array &G2,
          const Array &G3, const Array &G4, CplxVector &dW ) const;

      ///
      /// Construct the intrinsic differential operators on the surface mesh
      ///
      void constructDifferentialOperators();

      ///
      /// Construct the mesh function averaging operators
      ///
      void constructAveragingOperators();

      ///
      /// Assemble the various quantity lists into a single global vector
      ///
      /// Inputs:
      ///
      ///   muQ     #V by 1 complex representation of a quantitiy associated
      ///           with the Beltrami coefficient
      ///
      ///   w0Q     A complex variable associated with the complex parameter
      ///           of the Mobius mapping
      ///
      ///   phiQ    A scalar variable associated with the overall rotation
      ///           of the Mobius mapping
      ///
      /// Outputs:
      ///
      ///   xQ      (3 + 2*#V) by 1 real aggregation of values
      ///
      DSEM_INLINE void assembleGlobalQuantities (
          const CplxVector &muQ, const CScalar &w0Q, const Scalar &phiQ, Vector &xQ ) const;

      ///
      /// Disassemble a global vector into the various quantity lists
      ///
      /// Inputs:
      ///
      ///   xQ      (3 + 2*#V) by 1 real aggregation of values
      ///
      /// Outputs:
      ///
      ///   muQ     #V by 1 complex representation of a quantitiy associated
      ///           with the Beltrami coefficient
      ///
      ///   w0Q     A complex variable associated with the complex parameter
      ///           of the Mobius mapping
      ///
      ///   phiQ    A scalar variable associated with the overall rotation
      ///           of the Mobius mapping
      ///
      ///
      DSEM_INLINE void disassembleGlobalQuantities (
          const Vector &xQ, CplxVector &muQ, CScalar &w0Q, Scalar &phiQ ) const;

  };

}

#ifndef DSEM_STATIC_LIBRARY
#  include "DSEM.cpp"
#endif

#endif
