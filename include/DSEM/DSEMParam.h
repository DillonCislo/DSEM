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

#ifndef _DSEM_PARAM_H_
#define _DSEM_PARAM_H_

#include "dsemInline.h"
#include <Eigen/Core>

namespace DSEMpp {

  ///
  /// An enumeration of the different line search termination conditions
  ///
  enum LINE_SEARCH_TERMINATION_CONDITION {

    ///
    /// Accept any positive step size. NOT RECOMMENDED
    ///
    LINE_SEARCH_TERMINATION_NONE = 1,

    ///
    /// Find a step length that satisfies the sufficient decrease
    /// or Armijo condition.
    ///
    LINE_SEARCH_TERMINATION_ARMIJO = 2,

    ///
    /// Find a step length that satisfies the regular Wolfe conditions
    ///
    LINE_SEARCH_TERMINATION_WOLFE = 3,

    ///
    /// Find a step length that satisfies the strong Wolfe conditions
    ///
    LINE_SEARCH_TERMINATION_STRONG_WOLFE = 4,

  };

  ///
  /// An enumeration of the different line search methods
  ///
  enum LINE_SEARCH_METHOD {

    ///
    /// Simple backtracking
    ///
    LINE_SEARCH_BACKTRACKING = 1,

    ///
    /// A bracketing method. Similar to backtracking line search except that
    /// it actively maintains an upper and lower bound of the current search range
    ///
    LINE_SEARCH_BRACKETING = 2,

    ///
    /// An implementation of the line search algorithm by More and Thuente (1994)
    /// that satisfies the strong Wolfe conditions
    ///
    LINE_SEARCH_MORE_THUENTE = 3,

  };

  ///
  /// A class containing the parameters to control the BHF optimization algorithm
  /// BHF is a modified implementation of the L-BFGS algorithim.
  /// The structure of this class draws heavily from 'LBFGS++' by Yixuan Qiu
  /// 'https://github.com/yixuan/LBFGSpp'
  ///
  /// Template:
  ///
  ///   Scalar      The input type of data points and function values
  ///
  template <typename Scalar>
  class DSEMParam {

    private:

      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
      typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> ArrayVec;
      typedef Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> Array;
      typedef Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> CplxVector;
      typedef Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic> CplxMatrix;

    public:

      ///
      /// The number of corrections to approximate the inverse Hessian matrix.
      /// The L-BFGS routine stores the computation results of previous \ref m
      /// iterations to approximate the inverse Hessian matrix of the current
      /// iteration. This parameter controls the size of the limited memories
      /// (corrections). The default value is \c 6. Values less than \c 3 are
      /// not recommended. Large values will result in excessive computing time.
      ///
      int m;

      ///
      /// Absolute tolerance for convergence test.
      /// This parameter determines the absolute accuracy \f$\epsilon_{abs}\f$
      /// with which the solution is to be found. A minimization terminates when
      /// \f$||Pg||_{\infty} < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
      /// where \f$||x||\f$ denotes the Euclidean (L2) norm of \f$x\f$, and
      /// \f$Pg=P(x-g,l,u)-x\f$ is the projected gradient. The default value is
      /// \c 1e-5.
      ///
      Scalar epsilon;

      ///
      /// Relative tolerance for convergence test.
      /// This parameter determines the relative accuracy \f$\epsilon_{rel}\f$
      /// with which the solution is to be found. A minimization terminates when
      /// \f$||Pg||_{\infty} < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
      /// where \f$||x||\f$ denotes the Euclidean (L2) norm of \f$x\f$, and
      /// \f$Pg=P(x-g,l,u)-x\f$ is the projected gradient. The default value is
      /// \c 1e-5.
      ///
      Scalar epsilonRel;

      ///
      /// Distance for delta-based convergence test.
      /// This parameter determines the distance \f$d\f$ to compute the
      /// rate of decrease of the objective function,
      /// \f$f_{k-d}(x)-f_k(x)\f$, where \f$k\f$ is the current iteration
      /// step. If the value of this parameter is zero, the delta-based convergence
      /// test will not be performed. The default value is \c 1.
      ///
      int past;

      ///
      /// Delta for convergence test.
      /// The algorithm stops when the following condition is met,
      /// \f$|f_{k-d}(x)-f_k(x)|<\delta\cdot\max(1, |f_k(x)|, |f_{k-d}(x)|)\f$,
      /// where \f$f_k(x)\f$ is the current function value,
      /// and \f$f_{k-d}(x)\f$ is the function value
      /// \f$d\f$ iterations ago (specified by the \ref past parameter).
      /// The default value is \c 1e-10.
      ///
      Scalar delta;

      ///
      /// The maximum number of iterations.
      /// The optimization process is terminated when the iteration count
      /// exceeds this parameter. Setting this parameter to zero continues an
      /// optimization process until a convergence or error. The default value
      /// is \c 0.
      ///
      int maxIterations;

      ///
      /// The line search termination condition
      /// This parameter specifies the line search termination condition that
      /// will be used by the LBFGS routine. The default value is
      /// 'LINE_SEARCH_TERMINATION_STRONG_WOLFE'
      ///
      int lineSearchTermination;

      ///
      /// The line search method
      /// This parameter specifies the line search method that will be used
      /// by the LBFGS routine. The default value is 'LINE_SEARCH_BACKTRACKING'
      ///
      int lineSearchMethod;

      ///
      /// The maximum number of trials for the line search.
      /// This parameter controls the number of function and gradients evaluations
      /// per iteration for the line search routine. The default value is \c 20.
      ///
      int maxLineSearch;

      ///
      /// The minimum step length allowed in the line search.
      /// The default value is \c 1e-20. Usually this value does not need to be
      /// modified.
      ///
      Scalar minStep;

      ///
      /// The maximum step length allowed in the line search.
      /// The default value is \c 1e+20. Usually this value does not need to be
      /// modified.
      ///
      Scalar maxStep;

      ///
      /// A parameter to control the accuracy of the line search routine.
      /// The default value is \c 1e-4. This parameter should be greater
      /// than zero and smaller than \c 0.5.
      ///
      Scalar ftol;

      ///
      /// The coefficient for the Wolfe condition.
      /// This parameter is valid only when the line-search
      /// algorithm is used with the Wolfe condition.
      /// The default value is \c 0.9. This parameter should be greater
      /// the \ref ftol parameter and smaller than \c 1.0.
      ///
      Scalar wolfe;

      ///
      /// The coefficient of the term in the discrete energy that seeks
      /// to minimize the magnitude of the Beltrami coefficient corresponding
      /// to the minimum distance SEM (Conformality Coefficient). If this parameter
      /// is zero then this term is not included in the energy. The default
      /// value is zero.
      ///
      Scalar CC;

      ///
      /// The coefficient of the term in the discrete energy that seeks to
      /// minimize the magnitude of the gradient of the Beltrami coefficient
      /// corresponding to the minimum distance SEM (Smoothness Coefficient)
      /// If this parameter is zero then this term is not included in the
      /// energy. The default value is zero.
      ///
      Scalar SC;

      ///
      /// The coefficient of the inequality constraint terms in the discrete
      /// energy that keeps the magnitude of the Beltrami coefficient
      /// corresponding to the minimum distance SEM less than 1
      /// (Diffeomorphic Coefficient). If this parameter is zero then this
      /// term is not included in the energy. The default value is zero.
      ///
      Scalar DC;

    public:

      ///
      /// Constructor for parameter class
      ///
      DSEMParam();

      ///
      /// Check the validity of the parameters
      ///
      DSEM_INLINE void checkParam() const;

  };

}

#ifndef DSEM_STATIC_LIBRARY
# include "DSEMParam.cpp"
#endif

#endif

