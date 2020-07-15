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

#include "LineSearchBacktracking.h"
#include <stdexcept>
#include <iostream>
#include "../DSEM/clipToUnitCircle.h"

///
/// Line search by backtracking
///
template <typename Scalar, typename Index>
void DSEMpp::LineSearchBacktracking<Scalar, Index>::LineSearch(
    const DSEM<Scalar, Index> &dsem, const DSEMParam<Scalar> &param,
    const Vector &L, const Vector &tarA_E, const Vector &tarA_V,
    const Vector &xp, const CplxVector &wp,
    const Vector &drt, const CplxVector &dw,
    Scalar &fx, Vector &x, CplxVector &w,
    Vector &grad, Scalar &gradUNorm, Scalar &step, Vector &l ) {

  std::cout << "Check LSB0" << std::endl;

  // The number of faces
  int numF = dsem.m_F.rows();

  // The number of vertices
  int numV = dsem.m_V.rows();

  // Decreasing and increasing factors
  const Scalar dec = Scalar(0.5);
  const Scalar inc = Scalar(2.1);

  // Check the initial step length
  if ( step < Scalar(0.0) ) {
    std::invalid_argument("'step' must be positive");
  }

  // Save the function value at the current x
  const Scalar fx_init = fx;

  // Projection of the gradient onto the search direction
  const Scalar dg_init = grad.dot(drt);

  // Make sure the search direction is a descent direction
  if ( dg_init > Scalar(0.0) ) {
    throw std::logic_error("The update direction increases the objective function value");
  }

  const Scalar test_decr = param.ftol * dg_init;
  Scalar width;

  // The disassembled unknown lists
  CplxVector mu( numV, 1 );
  CplxVector muF( numF, 1);
  CScalar w0( Scalar(0.0), Scalar(0.0) );
  Scalar phi = Scalar(0.0);

  // The disassembled gradient lists
  CplxVector gradMu( numV, 1 );
  CScalar gradW0( Scalar(0.0), Scalar(0.0) );
  Scalar gradPhi = Scalar(0.0);

  // The quasiconformal mapping kernel components
  Array G1( numV, numV );
  Array G2( numV, numV );
  Array G3( numV, numV );
  Array G4( numV, numV );

  for( int iter = 0; iter < param.maxLineSearch; iter++ ) {

    std::cout << "Check LSB 1" << std::endl;

    // Evaluate the current candidate ---------------------------------------------------

    // x_{k+1} = x_k + step * d_k
    x.noalias() = xp + step * drt;

    // Disassemble updated global unknown vector
    dsem.disassembleGlobalQuantities( x, mu, w0, phi );

    // Update the quasiconformal mapping
    w.noalias() = wp + step * dw;

    // Clip the boundary points of the updated mapping to the unit circle
    DSEMpp::clipToUnitCircle( dsem.m_bdyIDx, w );

    std::cout << "Check LSB 2" << std::endl;

    // NOTE -----------------------------------------------------------------------------
    // This is a re-calculation of the Beltrami coefficient to avoid ringing
    // from the repeated averaging back and forth from vertices to faces.
    // It's more honest in a sense, but seems to have a severe negative effect on
    // convergence rates in test cases
    
    // Re-calculate the Beltrami coefficient from the mapping
    // muF = ( ( dsem.m_Dc * w ).array() / ( dsem.m_Dz * w ).array() ).matrix();
    // mu = dsem.m_F2V * muF;
    
    // Re-assemble the global unknown vector with the updated Beltrami coefficient
    // dsem.assembleGlobalQuantities( mu, w0, phi, x );
    
    // ----------------------------------------------------------------------------------

    std::cout << "Check LSB 3" << std::endl;
    // Calculate the updated mapping kernel
    dsem.calculateMappingKernel(w, G1, G2, G3, G4);

    std::cout << "Check LSB 4" << std::endl;

    // Evaluate the energy/gradients at the new location
    fx = dsem.calculateEnergyAndGrad( L, tarA_E, tarA_V,
        mu, w, phi, w0, G1, G2, G3, G4,
        gradMu, gradW0, gradPhi, gradUNorm, l );

    std::cout << "Check LSB 5" << std::endl;

    // Assemble the global gradient vector
    dsem.assembleGlobalQuantities( gradMu, gradW0, gradPhi, grad );

    // Evaluate line search termination conditions --------------------------------------
    
    // Accept any positive step
    if ( param.lineSearchTermination == LINE_SEARCH_TERMINATION_NONE ) {
      break;
    
    } else if (fx > fx_init + step * test_decr ) {

      width = dec;

    } else {

      // Armijo condition is met
      if ( param.lineSearchTermination == LINE_SEARCH_TERMINATION_ARMIJO ) {
        break;
      }

      const Scalar dg = grad.dot(drt);

      if ( dg < param.wolfe * dg_init ) {

        width = inc;

      } else {

        // Regular Wolfe condition is met
        if ( param.lineSearchTermination == LINE_SEARCH_TERMINATION_WOLFE ) {
          break;
        }

        if ( dg > -param.wolfe * dg_init ) {

          width = dec;

        } else {

          // Strong Wolfe condition is met
          break;

        }

      }

    }

    if ( iter >= param.maxLineSearch ) {

      throw std::runtime_error(
          "The line search routine reached the maximum number of iterations" );


    }

    if ( step < param.minStep ) {

      throw std::runtime_error(
          "The line search step became smaller than the minimum allowed value" );

    }

    if ( step > param.maxStep ) {

      throw std::runtime_error(
          "The line search step became larger than the maximum allowed value" );

    }

    step *= width;

    std::cout << "Check LSB 6" << std::endl;

  }

};

// TODO: Add explicit template instantiation
#ifdef DSEM_STATIC_LIBRARY
#endif
