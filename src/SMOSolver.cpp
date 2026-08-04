/*--------------------------------------------------------------------------*/
/*-------------------------- File SMOSolver.cpp ----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Implementation of the SMOSolver class.
 *
 * \author Donato Meoli \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 *
 * \copyright &copy; by Donato Meoli
 */
/*--------------------------------------------------------------------------*/
/*----------------------------- IMPLEMENTATION -----------------------------*/
/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include "SMOSolver.h"

#include <algorithm>

#include <cmath>

/*--------------------------------------------------------------------------*/
/*------------------------- NAMESPACE AND USING ----------------------------*/
/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

/*--------------------------------------------------------------------------*/
/*----------------------------- CONSTANTS ----------------------------------*/
/*--------------------------------------------------------------------------*/

/// the curvature substituted for a nonpositive one, i.e., for a kernel
/// matrix that is not positive definite
static constexpr double dTau = 1e-12;

/// the tolerance within which a multiplier is snapped to one of its bounds
static constexpr double dSnap = 1e-12;

/*--------------------------------------------------------------------------*/
/*----------------------------- STATIC MEMBERS -----------------------------*/
/*--------------------------------------------------------------------------*/

// register SMOSolver in the Solver factory

SMSpp_insert_in_factory_cpp_1( SMOSolver );

/*--------------------------------------------------------------------------*/
/*--------------------- DERIVED METHODS OF BASE CLASS ----------------------*/
/*--------------------------------------------------------------------------*/

void SMOSolver::set_Block( Block * block )
{
 if( block == f_Block )  // actually doing nothing
  return;                // cowardly and silently return

 Solver::set_Block( block );

 f_solved = false;
 f_value = 0;
 f_b = 0;
 f_iter = 0;
 v_alpha.clear();
 v_G.clear();

 if( ! block ) {
  f_SVM = nullptr;
  return;
  }

 f_SVM = dynamic_cast< SVMBlock * >( block );
 if( ! f_SVM )
  throw( std::invalid_argument( "SMOSolver::set_Block: the Block is not a "
                                "SVMBlock" ) );

 }  // end( SMOSolver::set_Block )

/*--------------------------------------------------------------------------*/

int SMOSolver::compute( bool changedvars )
{
 if( ! f_SVM )
  throw( std::logic_error( "SMOSolver::compute: no SVMBlock is set" ) );

 lock();  // lock the mutex

 /* Every call re-reads the whole data of the dual out of the SVMBlock and
  * restarts from the origin, hence whatever changed since the previous call
  * is taken into account by construction and the queued Modification can just
  * be dropped. */
 v_mod.clear();

 f_solved = false;
 f_iter = 0;

 // cache the data of the dual out of the SVMBlock- - - - - - - - - - - - - -

 f_n = f_SVM->get_NSamples();
 f_N = f_SVM->get_NDual();
 f_u = f_SVM->get_ub();
 f_rb = f_SVM->get_reg_bias() ? 1 : 0;
 f_d = f_SVM->get_squared_loss() ? 1 / ( 2 * f_SVM->get_C() ) : 0;

 f_K = f_SVM->get_K().data();
 f_ds = f_SVM->get_dual_signs().data();
 f_di = f_SVM->get_dual_samples().data();
 f_dq = f_SVM->get_dual_costs().data();

 // start from the origin, where the gradient is just the linear term - - - -

 v_alpha.assign( f_N , 0 );
 v_G.assign( f_dq , f_dq + f_N );

 int status;
 if( f_rb ) {
  status = solve_box();

  // with the bias regularised there is no equality constraint, and the bias
  // is just one more component of the weight vector
  f_b = 0;
  for( Index k = 0 ; k < f_N ; ++k )
   f_b += f_ds[ k ] * v_alpha[ k ];
  }
 else
  status = solve_with_equality();

 // the value of the dual at the solution: 1/2 alpha^T Q alpha + q^T alpha =
 // 1/2 alpha^T ( G + q ), since G = Q alpha + q
 double v = 0;
 for( Index k = 0 ; k < f_N ; ++k )
  v += v_alpha[ k ] * ( v_G[ k ] + f_dq[ k ] );
 f_value = v / 2;

 /* What is reported has to be the value of the Objective of the SVMBlock,
  * which is the dual only if that is the formulation its abstract
  * representation encodes: for the primal ones it is the opposite, strong
  * duality holding since the training problem is convex. */
 const int form = f_SVM->get_generated_formulation();
 if( ( form == SVMBlock::kPrimal ) || ( form == SVMBlock::kDecomposed ) )
  f_value = - f_value;

 f_solved = true;

 unlock();  // unlock the mutex

 return( status );

 }  // end( SMOSolver::compute )

/*--------------------------------------------------------------------------*/

void SMOSolver::get_var_solution( Configuration * solc )
{
 if( ! f_solved )
  throw( std::logic_error( "SMOSolver::get_var_solution: no solution is "
                           "available" ) );

 auto alpha = v_alpha;  // the Solver keeps its own copy for a re-solve
 f_SVM->set_dual_solution( std::move( alpha ) , f_b );

 // if the abstract representation of the dual is there, fill it in as well
 if( auto av = f_SVM->get_static_variable_v< ColVariable >( "alpha" ) )
  for( Index k = 0 ; k < f_N ; ++k )
   (*av)[ k ].set_value( v_alpha[ k ] );

 }  // end( SMOSolver::get_var_solution )

/*--------------------------------------------------------------------------*/
/*--------------------------- PROTECTED METHODS ----------------------------*/
/*--------------------------------------------------------------------------*/

int SMOSolver::solve_with_equality( void )
{
 /* The optimality conditions of the dual are
  *
  *   max { g_i : i in I_up } <= min { g_j : j in I_low }
  *
  * with g_k = - s_k G_k, where I_up (I_low) is the set of the multipliers
  * that can be increased (decreased) along the sign s_k without violating
  * their bounds, i.e., the ones that can be moved in a direction that keeps
  * s^T alpha = 0. The common value of the two sides at optimality is the
  * multiplier of the equality constraint, i.e., the bias of the model. */

 while( ( f_max_iter < 0 ) || ( f_iter < Index( f_max_iter ) ) ) {

  // select the maximal violating pair - - - - - - - - - - - - - - - - - - -

  Index i = f_N , j = f_N;
  double m = - Inf< double >() , M = Inf< double >();

  for( Index k = 0 ; k < f_N ; ++k ) {
   const double sk = f_ds[ k ];
   const double ak = v_alpha[ k ];
   const double gk = - sk * v_G[ k ];

   const bool up = ( sk > 0 ) ? ( ak < f_u ) : ( ak > 0 );
   const bool low = ( sk > 0 ) ? ( ak > 0 ) : ( ak < f_u );

   if( up && ( gk > m ) ) { m = gk; i = k; }
   if( low && ( gk < M ) ) { M = gk; j = k; }
   }

  if( ( i == f_N ) || ( j == f_N ) )  // no feasible direction exists at all
   return( kOK );                     // hence the point is optimal

  // at optimality the two sides coincide with the multiplier of the equality
  // constraint, i.e., with the bias; in general it is any value in between
  f_b = ( m + M ) / 2;

  if( m - M <= f_tol )  // the optimality conditions hold
   return( kOK );

  // minimize along the only feasible direction changing alpha_i, alpha_j - -

  const double si = f_ds[ i ] , sj = f_ds[ j ];

  // the curvature along the direction, which for the plain classification
  // dual is K_ii + K_jj - 2 K_ij
  double a = Q( i , i ) + Q( j , j ) - 2 * si * sj * Q( i , j );

  /* A nonpositive curvature means that the kernel does not obey Mercer's
   * condition, in which case the objective is concave along the direction and
   * its minimum over the segment is at one of the two ends. Since the pair is
   * selected so that the derivative at t = 0 is negative, that end is the
   * farthest reachable one, which is what substituting a tiny curvature
   * yields. */
  if( a <= 0 )
   a = dTau;

  double t = ( m - M ) / a;  // the unconstrained minimizer along t

  // clip t so that both multipliers stay within their bounds
  const double ti = ( si > 0 ) ? f_u - v_alpha[ i ] : v_alpha[ i ];
  const double tj = ( sj > 0 ) ? v_alpha[ j ] : f_u - v_alpha[ j ];
  t = std::min( t , std::min( ti , tj ) );

  if( t <= 0 )  // no progress is possible: numerically optimal
   return( kOK );

  const double dai = si * t;
  const double daj = - sj * t;

  v_alpha[ i ] += dai;
  v_alpha[ j ] += daj;

  // snap to the bounds to keep the sets I_up and I_low exact
  if( v_alpha[ i ] < dSnap ) v_alpha[ i ] = 0;
  if( v_alpha[ j ] < dSnap ) v_alpha[ j ] = 0;
  if( f_u < Inf< double >() ) {
   if( v_alpha[ i ] > f_u - dSnap ) v_alpha[ i ] = f_u;
   if( v_alpha[ j ] > f_u - dSnap ) v_alpha[ j ] = f_u;
   }

  // update the gradient - - - - - - - - - - - - - - - - - - - - - - - - - -

  const double * Ki = f_K + std::size_t( f_di[ i ] ) * f_n;
  const double * Kj = f_K + std::size_t( f_di[ j ] ) * f_n;

  for( Index k = 0 ; k < f_N ; ++k )
   v_G[ k ] += f_ds[ k ] * ( si * ( Ki[ f_di[ k ] ] + f_rb ) * dai +
                             sj * ( Kj[ f_di[ k ] ] + f_rb ) * daj );

  if( f_d ) {  // the diagonal term is not part of the kernel expansion
   v_G[ i ] += f_d * dai;
   v_G[ j ] += f_d * daj;
   }

  ++f_iter;
  }

 return( kStopIter );

 }  // end( SMOSolver::solve_with_equality )

/*--------------------------------------------------------------------------*/

int SMOSolver::solve_box( void )
{
 /* Without the equality constraint the multipliers are independent, hence
  * one of them at a time is moved to the minimizer of the dual along its own
  * coordinate. The multiplier with the largest projected gradient is chosen,
  * and the algorithm stops when even that one is small enough. */

 while( ( f_max_iter < 0 ) || ( f_iter < Index( f_max_iter ) ) ) {

  Index i = f_N;
  double viol = 0;

  for( Index k = 0 ; k < f_N ; ++k ) {
   const double ak = v_alpha[ k ];
   double pg = v_G[ k ];
   if( ( ak <= 0 ) && ( pg > 0 ) )
    pg = 0;
   else if( ( ak >= f_u ) && ( pg < 0 ) )
    pg = 0;

   if( std::abs( pg ) > viol ) { viol = std::abs( pg ); i = k; }
   }

  if( ( i == f_N ) || ( viol <= f_tol ) )  // the optimality conditions hold
   return( kOK );

  double a = Q( i , i );
  if( a <= 0 )
   a = dTau;

  double ai = v_alpha[ i ] - v_G[ i ] / a;
  ai = std::max( double( 0 ) , std::min( ai , f_u ) );

  const double dai = ai - v_alpha[ i ];
  if( dai == 0 )  // no progress is possible: numerically optimal
   return( kOK );

  v_alpha[ i ] = ai;

  const double si = f_ds[ i ];
  const double * Ki = f_K + std::size_t( f_di[ i ] ) * f_n;

  for( Index k = 0 ; k < f_N ; ++k )
   v_G[ k ] += f_ds[ k ] * si * ( Ki[ f_di[ k ] ] + f_rb ) * dai;

  if( f_d )
   v_G[ i ] += f_d * dai;

  ++f_iter;
  }

 return( kStopIter );

 }  // end( SMOSolver::solve_box )

/*--------------------------------------------------------------------------*/
/*------------------------- End File SMOSolver.cpp -------------------------*/
/*--------------------------------------------------------------------------*/
