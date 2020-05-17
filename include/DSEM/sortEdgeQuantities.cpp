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

#include "sortEdgeQuantities.h"

#include <cassert>
#include <stdexcept>

#include <igl/sort.h>
#include <igl/ismember.h>

///
/// Sort a vector of quantities based on the ordering of an initial and
/// target edge connectivity correspondence
///
template <typename Scalar, typename Index>
DSEM_INLINE void DSEMpp::sortEdgeQuantities(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &Q,
    const Eigen::Matrix<Index, Eigen::Dynamic, 2> &E1,
    const Eigen::Matrix<Index, Eigen::Dynamic, 2> &E2,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &QS ) {

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 2> EdgeVector;
  typedef Eigen::Matrix<Index, Eigen::Dynamic, 1> IndexVector;
  typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> BoolVector;

  assert( (Q.rows() == E1.rows()) && (Q.rows() == E2.rows()) &&
      "Invalid edge quantity correspondence" );

  // The number of edges
  int numE = Q.size();

  // Sort the columns of each edge connectivity list
  EdgeVector E1SC(numE, 2);
  igl::sort(E1, 2, true, E1SC);

  EdgeVector E2SC(numE, 2);
  igl::sort(E2, 2, true, E2SC);

  // Find the edge list correspondence
  BoolVector IA(numE, 1);
  IndexVector LOCB(numE, 1);
  igl::ismember_rows(E2SC, E1SC, IA, LOCB);

  // Update output quantitiy list
  QS.resizeLike(Q);
  
  for( int i = 0; i < numE; i++ ) {
    QS(i) = Q(LOCB(i));
  }

};

///
/// Sort a vector of quantities based on the ordering of an initial and
/// target edge connectivity correspondence
///
template <typename Scalar, typename Index>
DSEM_INLINE void DSEMpp::sortEdgeQuantities(
    const Eigen::Matrix<Index, Eigen::Dynamic, 2> &E1,
    const Eigen::Matrix<Index, Eigen::Dynamic, 2> &E2,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &Q ) {

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 2> EdgeVector;
  typedef Eigen::Matrix<Index, Eigen::Dynamic, 1> IndexVector;
  typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> BoolVector;

  assert( (Q.rows() == E1.rows()) && (Q.rows() == E2.rows()) &&
      "Invalid edge quantity correspondence" );

  // The number of edges
  int numE = Q.size();
  
  // Sort the columns of each edge connectivity list
  EdgeVector E1SC(numE, 2);
  igl::sort(E1, 2, true, E1SC);

  EdgeVector E2SC(numE, 2);
  igl::sort(E2, 2, true, E2SC);

  // Find the edge list correspondence
  BoolVector IA(numE, 1);
  IndexVector LOCB(numE, 1);
  igl::ismember_rows(E2SC, E1SC, IA, LOCB);

  Vector Qtmp = Q;
  for( int i = 0; i < numE; i++ ) {
    Q(i) = Qtmp(LOCB(i));
  }

};

// TODO: Add explicit template instantiation
#ifdef DSEM_STATIC_LIBRARY
#endif





