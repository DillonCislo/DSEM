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

#ifndef _SORT_EDGE_QUANTITIES_H_
#define _SORT_EDGE_QUANTITIES_H_

#include "dsemInline.h"
#include <Eigen/Core>

namespace DSEMpp {

  ///
  /// Sort a vector of quantities based on the ordering of an initial and
  /// target edge connectivity corresponence
  ///
  ///  Template:
  ///
  ///   Scalar    Input type of data
  ///   Index     Input type of indices
  ///
  ///  Inputs:
  ///
  ///   Q     #Q by 1 list of edge based scalar quantities
  ///   E1    #E by 1 initial edge connectivity list
  ///   E2    #E by 1 target edge connectivity list
  ///
  /// Outputs:
  ///
  ///   QS    #Q by 1 sorted list of edge based quantities
  ///
  template <typename Scalar, typename Index>
  DSEM_INLINE void sortEdgeQuantities(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &Q,
      const Eigen::Matrix<Index, Eigen::Dynamic, 2> &E1,
      const Eigen::Matrix<Index, Eigen::Dynamic, 2> &E2,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &QS );

  ///
  /// Sort a vector of quantities based on the ordering of an initial and
  /// target edge connectivity corresponence
  ///
  ///  Template:
  ///
  ///   Scalar    Input type of data
  ///   Index     Input type of indices
  ///
  ///  Inputs:
  ///
  ///   Q     #Q by 1 list of edge based scalar quantities
  ///   E1    #E by 1 initial edge connectivity list
  ///   E2    #E by 1 target edge connectivity list
  ///
  /// Outputs:
  ///
  ///   Q    #Q by 1 sorted list of edge based quantities
  ///
  template <typename Scalar, typename Index>
  DSEM_INLINE void sortEdgeQuantities(
      const Eigen::Matrix<Index, Eigen::Dynamic, 2> &E1,
      const Eigen::Matrix<Index, Eigen::Dynamic, 2> &E2,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &Q );

};

#ifndef DSEM_STATIC_LIBRARY
#  include "sortEdgeQuantities.cpp"
#endif

#endif
