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

#include <map>

#include <numeric>

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

 /* Find out what has changed since the previous call: a warm start is only
  * possible if every queued Modification allows it, and there is nothing to
  * start from unless a solution has been found already. */

 bool warm = f_solved;

 for( ; ; ) {
  auto mod = pop();
  if( ! mod )
   break;
  if( ! guts_of_poM( mod.get() ) )
   warm = false;
  }

 // realign to the SVMBlock, from the previous solution or from the origin - -

 if( ( ! warm ) || ( ! resync() ) )
  reload();

 f_solved = false;
 f_iter = 0;

 int status;
 if( f_rb ) {
  status = solve_box();

  /* With the bias regularised there is no equality constraint, and the bias
   * is just one more component of the weight vector, whence the division by
   * the weight of the regularisation term and the linear term of the primal
   * [see SVMBlock::set_linear_term()]. */
  double sa = 0;
  for( Index k = 0 ; k < f_N ; ++k )
   sa += f_ds[ k ] * v_alpha[ k ];

  f_b = ( sa - f_SVM->get_linear_bias() ) / f_rw;
  }
 else {
  /* The multipliers have to satisfy the equality constraint before the SMO
   * iteration starts, since it keeps s^T alpha where it finds it: they do,
   * unless its right-hand side has just changed or they come from the
   * origin. */
  if( ! restore_equality() ) {
   f_solved = false;
   unlock();
   return( kUnbounded );
   }

  status = solve_with_equality();
  }

 // the value of the dual at the solution: 1/2 alpha^T Q alpha + q^T alpha =
 // 1/2 alpha^T ( G + q ), since G = Q alpha + q
 double v = 0;
 for( Index k = 0 ; k < f_N ; ++k )
  v += v_alpha[ k ] * ( v_G[ k ] + f_dq[ k ] );
 /* The expression above is the one the SMO iteration minimises, i.e., the
  * opposite of the Wolfe dual objective; what has to be reported is the value
  * of the Objective of the SVMBlock, which is the optimal value of the
  * training problem whichever formulation it encodes, the Wolfe dual being
  * written as the maximisation whose value strong duality makes equal to the
  * primal's. Hence the sign, once and for all, with no dependence on what the
  * abstract representation happens to be. */
 f_value = - v / 2 + f_dc;

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

 // the model is now in the SVMBlock; leave it also where any Solver working
 // on the abstract representation would have left it, if there is one
 f_SVM->set_solution_in_abstract();

 }  // end( SMOSolver::get_var_solution )

/*--------------------------------------------------------------------------*/

Solution * SMOSolver::get_Solution( Configuration * solc )
{
 if( ! f_solved )
  return( nullptr );

 /* Which Solution the Configuration asks for is the SVMBlock's business, and
  * asking it for an empty one is how it is found out: filling it is this
  * Solver's business only if it is the one saving the model, which is
  * precisely the one that requires no Variable to exist. */

 auto sol = f_SVM->get_Solution( solc , true );

 if( auto msol = dynamic_cast< SVMBlockSolution * >( sol ) ) {
  // the multipliers are copied, since the Solver keeps its own for a re-solve
  msol->set_dual_model( doubleVec( v_alpha ) , f_b );
  return( msol );
  }

 delete sol;

 /* Anything else saves the abstract representation, which this Solver does
  * not write into: the base class does the only thing that can be done, that
  * is, write the solution into the Variable and ask the SVMBlock for it. */

 return( Solver::get_Solution( solc ) );

 }  // end( SMOSolver::get_Solution )

/*--------------------------------------------------------------------------*/
/*--------------------------- PROTECTED METHODS ----------------------------*/
/*--------------------------------------------------------------------------*/

bool SMOSolver::guts_of_poM( const Modification * mod )
{
 // a GroupModification allows a warm start only if all of its members do
 if( auto gm = dynamic_cast< const GroupModification * >( mod ) ) {
  bool ok = true;
  for( const auto & sm : gm->sub_Modifications() )
   if( ! guts_of_poM( sm.get() ) )
    ok = false;
  return( ok );
  }

 if( dynamic_cast< const NBModification * >( mod ) )
  return( false );  // everything has changed

 if( auto sm = dynamic_cast< const SVMBlockMod * >( mod ) )
  switch( sm->type() ) {
   case( SVMBlockMod::eChgKernel ):
   case( SVMBlockMod::eChgRegBias ):
    return( false );  // the Hessian of the dual changes as a whole
   case( SVMBlockMod::eAddSamples ):
   case( SVMBlockMod::eRmvSamples ):
    /* The dual index space changes size, but the multipliers of the samples
     * that are still there are still worth starting from: which they are is
     * what the sample map says. */
    compose_smap( mod );
    return( true );
   default:
    return( true );   // only the bounds, the linear term or the diagonal do
   }

 /* An abstract Modification issued by the SVMBlock while changing its own
  * physical representation describes something that the physical Modification
  * issued alongside it has already accounted for. Anything else means that
  * someone has changed the abstract representation directly: since this
  * Solver does not read it, and the physical representation may therefore no
  * longer agree with it, there is nothing better to do than to start over. */

 return( ! mod->concerns_Block() );

 }  // end( SMOSolver::guts_of_poM )

/*--------------------------------------------------------------------------*/

void SMOSolver::compose_smap( const Modification * mod )
{
 /* The map goes from the samples of the data set this Solver is aligned to
  * to those of the current one, so the *first* change starts it as the
  * identity and every other one is composed onto it. */

 if( v_smap.empty() ) {
  v_smap.resize( f_n );
  std::iota( v_smap.begin() , v_smap.end() , Index( 0 ) );
  }

 if( auto rm = dynamic_cast< const SVMBlockRngdMod * >( mod ) ) {
  // samples appended at the end: they were not there before
  const auto rng = rm->rng();
  v_smap.resize( v_smap.size() + ( rng.second - rng.first ) ,
                 Inf< Index >() );
  return;
  }

 if( auto sm = dynamic_cast< const SVMBlockSbstMod * >( mod ) ) {
  // samples removed: the survivors keep their order, hence so does the map
  auto & nms = sm->nms();
  Subset nmap;
  nmap.reserve( v_smap.size() - nms.size() );

  auto rmv = nms.begin();
  for( Index i = 0 ; i < v_smap.size() ; ++i )
   if( ( rmv != nms.end() ) && ( *rmv == i ) )
    ++rmv;
   else
    nmap.push_back( v_smap[ i ] );

  v_smap = std::move( nmap );
  }

 }  // end( SMOSolver::compose_smap )

/*--------------------------------------------------------------------------*/

void SMOSolver::reload( void )
{
 f_n = f_SVM->get_NSamples();
 f_N = f_SVM->get_NDual();
 f_u = f_SVM->get_ub();
 f_rb = f_SVM->get_reg_bias() ? 1 : 0;
 f_d = f_SVM->get_squared_loss() ? 1 / ( 2 * f_SVM->get_C() ) : 0;
 f_rw = f_SVM->get_reg_weight();

 f_K = f_SVM->get_K().data();
 f_di = f_SVM->get_dual_samples().data();

 v_s = f_SVM->get_dual_signs();
 v_di_c = f_SVM->get_dual_samples();

 /* The linear term of the primal shifts the linear coefficients of the dual,
  * moves the right-hand side of the equality constraint and adds a constant
  * to the value [see SVMBlock::set_linear_term()]: it is folded into the data
  * of the dual once and for all here, so that the algorithm need not know
  * about it. */
 v_q = f_SVM->get_dual_costs();
 { doubleVec shift;
   f_SVM->get_dual_shifts( shift );
   for( Index k = 0 ; k < f_N ; ++k )
    v_q[ k ] -= shift[ k ];
   }

 f_mu = f_rb ? 0 : f_SVM->get_linear_bias();
 f_dc = f_SVM->get_dual_constant();
 f_ds = v_s.data();
 f_dq = v_q.data();

 v_smap.clear();   // whatever has changed, this is a fresh start

 // start from the origin, where the gradient is just the linear term
 v_alpha.assign( f_N , 0 );
 v_G = v_q;

 fill_diagonal();

 }  // end( SMOSolver::reload )

/*--------------------------------------------------------------------------*/

bool SMOSolver::resample( void )
{
 /* The dual index space has changed size, samples having been added or
  * removed: each new dual index is matched with the old one that referred to
  * the same sample with the same sign, since that is all that identifies it
  * [see the comments to SVMBlock]. The multiplier of a dual index that
  * survives is kept, that of a new one starts at zero. */

 const Index N = f_SVM->get_NDual();
 auto & di = f_SVM->get_dual_samples();
 auto & s = f_SVM->get_dual_signs();
 auto & q = f_SVM->get_dual_costs();

 if( ( di.size() != N ) || ( v_smap.size() != f_SVM->get_NSamples() ) )
  return( false );

 std::map< std::pair< Index , bool > , Index > o_k;
 for( Index k = 0 ; k < v_di_c.size() ; ++k )
  o_k[ { v_di_c[ k ] , v_s[ k ] > 0 } ] = k;

 doubleVec n_alpha( N , 0 );
 for( Index k = 0 ; k < N ; ++k ) {
  const Index i = v_smap[ di[ k ] ];
  if( i == Inf< Index >() )   // a new sample: its multiplier starts at zero
   continue;
  auto it = o_k.find( { i , s[ k ] > 0 } );
  if( it != o_k.end() )
   n_alpha[ k ] = v_alpha[ it->second ];
  }

 // the data of the dual, which is the new one from here on
 f_n = f_SVM->get_NSamples();
 f_N = N;
 f_u = f_SVM->get_ub();
 f_d = f_SVM->get_squared_loss() ? 1 / ( 2 * f_SVM->get_C() ) : 0;
 f_rw = f_SVM->get_reg_weight();
 f_K = f_SVM->get_K().data();
 f_di = di.data();
 v_s = s;
 v_q = q;
 v_di_c = di;
 f_ds = v_s.data();
 f_dq = v_q.data();
 v_alpha = std::move( n_alpha );
 v_smap.clear();

 /* Removing a sample whose multiplier was nonzero leaves the equality
  * constraint violated, and SMO starts from a feasible point: the excess is
  * given back to the multipliers that can absorb it, which keeps the start
  * as close as possible to the previous solution instead of throwing it
  * away. Nothing to do without the equality constraint, i.e., with the bias
  * regularised. */

 if( ! f_rb ) {
  double delta = 0;
  for( Index k = 0 ; k < f_N ; ++k )
   delta += v_s[ k ] * v_alpha[ k ];

  for( Index k = 0 ; ( k < f_N ) && ( std::abs( delta ) > dSnap ) ; ++k ) {
   // how much this multiplier can move in the direction that reduces the
   // violation, i.e., down if its sign agrees with the excess and up if not
   const double room = ( ( v_s[ k ] > 0 ) == ( delta > 0 ) )
                       ? v_alpha[ k ] : f_u - v_alpha[ k ];
   if( room <= 0 )
    continue;

   const double step = std::min( room , std::abs( delta ) );
   v_alpha[ k ] += ( ( v_s[ k ] > 0 ) == ( delta > 0 ) ) ? - step : step;
   delta -= ( ( v_s[ k ] > 0 ) == ( delta > 0 ) ) ? v_s[ k ] * step
                                                  : - v_s[ k ] * step;
   }

  if( std::abs( delta ) > dSnap )   // it could not be absorbed
   return( false );                 // start over
  }

 /* The gradient is recomputed rather than updated: a change of the dual
  * index space touches every entry, and one pass over the Gram matrix, which
  * is cached and has just been extended or compacted rather than recomputed,
  * costs a fraction of the sweeps the re-optimization would take anyway. */

 v_G = v_q;
 for( Index k = 0 ; k < f_N ; ++k ) {
  const double ak = v_alpha[ k ];
  if( ! ak )
   continue;
  for( Index l = 0 ; l < f_N ; ++l )
   v_G[ l ] += Q( l , k ) * ak;
  }

 return( true );

 }  // end( SMOSolver::resample )

/*--------------------------------------------------------------------------*/

bool SMOSolver::resync( void )
{
 if( ( v_alpha.size() != f_N ) || ( v_G.size() != f_N ) ||
     ( v_s.size() != f_N ) || ( v_q.size() != f_N ) ||
     ( f_rb != ( f_SVM->get_reg_bias() ? 1 : 0 ) ) )
  return( false );

 // samples have been added or removed: the multipliers of those that are
 // still there are realigned to the new dual index space, and the gradient
 // is recomputed from them
 if( ! v_smap.empty() ) {
  if( ! resample() )
   return( false );
  }
 else {
  // the size and the structure of the dual have to be the same
  if( ( f_N != f_SVM->get_NDual() ) || ( f_n != f_SVM->get_NSamples() ) )
   return( false );

  // and so have the signs, which are all over the Hessian
  auto & s = f_SVM->get_dual_signs();
  if( ! std::equal( s.begin() , s.end() , v_s.begin() ) )
   return( false );
  }

 f_K = f_SVM->get_K().data();  // both may have been moved elsewhere in the
 f_di = f_SVM->get_dual_samples().data();            // meantime

 /* The gradient G = Q alpha + q is affine in the multipliers, in the linear
  * term and in the diagonal alike, whence each of the three changes below is
  * followed exactly, and in O( N ) time. They are applied in the order in
  * which they compose: the scaling uses the previous linear term, and the
  * diagonal the scaled multipliers. */

 const double u = f_SVM->get_ub();

 if( u < Inf< double >() ) {
  double mx = 0;
  for( auto ak : v_alpha )
   mx = std::max( mx , ak );

  if( mx > u ) {
   /* The multipliers are scaled, rather than clipped, so that they keep
    * satisfying the equality constraint, which is homogeneous; scaling them
    * by rho gives the gradient rho ( G - q ) + q. */
   const double rho = u / mx;
   for( Index k = 0 ; k < f_N ; ++k ) {
    v_G[ k ] = rho * ( v_G[ k ] - v_q[ k ] ) + v_q[ k ];
    v_alpha[ k ] *= rho;
    }
   }
  }

 f_u = u;

 /* The weight of the regularisation term divides the Hessian, hence changing
  * it scales the part of the gradient that comes from it: G - q - d alpha is
  * the Gram part, which is what gets rescaled. */

 const double rw = f_SVM->get_reg_weight();

 if( rw != f_rw ) {
  const double rho = f_rw / rw;
  for( Index k = 0 ; k < f_N ; ++k ) {
   const double dk = f_d * v_alpha[ k ];
   v_G[ k ] = rho * ( v_G[ k ] - v_q[ k ] - dk ) + dk + v_q[ k ];
   }
  f_rw = rw;
  }

 auto & q = f_SVM->get_dual_costs();

 doubleVec shift;
 f_SVM->get_dual_shifts( shift );

 for( Index k = 0 ; k < f_N ; ++k ) {
  const double qk = q[ k ] - shift[ k ];
  if( qk != v_q[ k ] ) {
   v_G[ k ] += qk - v_q[ k ];
   v_q[ k ] = qk;
   }
  }

 f_mu = f_rb ? 0 : f_SVM->get_linear_bias();
 f_dc = f_SVM->get_dual_constant();

 const double d = f_SVM->get_squared_loss() ? 1 / ( 2 * f_SVM->get_C() ) : 0;

 if( d != f_d ) {
  for( Index k = 0 ; k < f_N ; ++k )
   v_G[ k ] += ( d - f_d ) * v_alpha[ k ];
  f_d = d;
  }

 fill_diagonal();   // it depends on all of the above

 return( true );

 }  // end( SMOSolver::resync )

/*--------------------------------------------------------------------------*/

bool SMOSolver::restore_equality( void )
{
 double sa = 0;
 for( Index k = 0 ; k < f_N ; ++k )
  sa += f_ds[ k ] * v_alpha[ k ];

 double diff = f_mu - sa;   // how much s^T alpha has to change

 const double eps = 1e-10 * ( 1 + std::abs( f_mu ) );

 /* One multiplier at a time is moved as far as its bounds allow in the
  * direction that reduces the residual, which takes the fewest of them that
  * can do it; the gradient is updated along with each of them, exactly as
  * the SMO step does. Closing the residual with the multiplier k costs
  * G_k s_k diff to the first order, whence the one that is moved is the one
  * for which this is smallest: this is what keeps the point the iteration
  * starts from a good one, and it is all the more so when the multipliers
  * are unbounded above, in which case any one of them could do it alone. */

 while( std::abs( diff ) > eps ) {
  Index h = f_N;
  double best = Inf< double >();

  for( Index k = 0 ; k < f_N ; ++k ) {
   const double sk = f_ds[ k ];
   const double room = ( diff * sk > 0 ) ? f_u - v_alpha[ k ] : v_alpha[ k ];
   if( room <= 0 )
    continue;

   const double cost = v_G[ k ] * sk * diff;
   if( cost < best ) { best = cost; h = k; }
   }

  if( h == f_N )   // no multiplier can be moved any further
   return( false );

  const double sh = f_ds[ h ];
  double t = diff * sh;   // the move that would close the residual at once

  t = std::max( - v_alpha[ h ] , std::min( t , f_u - v_alpha[ h ] ) );

  if( ! t )   // the room there is is below the precision of the sum
   return( false );

  v_alpha[ h ] += t;
  diff -= sh * t;

  const double * Kh = f_K + std::size_t( f_di[ h ] ) * f_n;
  const double dh = sh * t / f_rw;

  for( Index l = 0 ; l < f_N ; ++l )
   v_G[ l ] += f_ds[ l ] * ( Kh[ f_di[ l ] ] + f_rb ) * dh;

  if( f_d )   // the diagonal term is not part of the kernel expansion
   v_G[ h ] += f_d * t;
  }

 return( true );

 }  // end( SMOSolver::restore_equality )

/*--------------------------------------------------------------------------*/

void SMOSolver::fill_diagonal( void )
{
 /* The diagonal of the Hessian of the dual, which the selection of the
  * working set reads once per candidate: taking it out of the Gram matrix
  * every time would walk it with stride n + 1, i.e. one cache miss per
  * candidate, which is what makes the second order selection expensive. */

 v_QD.resize( f_N );

 for( Index k = 0 ; k < f_N ; ++k ) {
  const std::size_t ik = std::size_t( f_di[ k ] );
  v_QD[ k ] = ( f_K[ ik * f_n + ik ] + f_rb ) / f_rw + f_d;
  }

 }  // end( SMOSolver::fill_diagonal )

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
  * s^T alpha where it is. The common value of the two sides at optimality is
  * the multiplier of the equality constraint, i.e., the bias of the model.
  *
  * The pair that is moved is selected with *second order* information: the
  * first of the two is the maximal violating index, as the pair of Platt is,
  * but the second one is the index that gives the largest decrease of the
  * objective, i.e.,
  *
  *   max { ( g_i - g_k )^2 / a_ik : k in I_low , g_k < g_i }
  *
  * with a_ik the curvature along the direction, rather than the minimal
  * violating one. The two coincide only if the curvature is the same in
  * every direction, and the difference is worth the O( N ) it costs, which
  * is what the maximal violating pair costs anyway.
  *
  * Every so often the multipliers that are at a bound and cannot be selected
  * are taken out of the active set [see shrink()], so that the iterations,
  * which cost O( | A | ), become cheaper as the solution settles; when the
  * conditions hold on the active set they are checked on the whole of it
  * [see unshrink()], and the algorithm only stops if they hold there too. */

 while( ( f_max_iter < 0 ) || ( f_iter < Index( f_max_iter ) ) ) {

  // select the first index of the pair: the maximal violating one - - - - -

  Index i = f_N;
  double m = - Inf< double >();

  for( Index k = 0 ; k < f_N ; ++k ) {
   const double sk = f_ds[ k ];
   const double gk = - sk * v_G[ k ];
   const bool up = ( sk > 0 ) ? ( v_alpha[ k ] < f_u ) : ( v_alpha[ k ] > 0 );

   if( up && ( gk > m ) ) { m = gk; i = k; }
   }

  /* Select the second one: the index of I_low that gives the largest
   * decrease among those that make a violating pair with the first. The
   * minimal violating index is computed along the way, since it is what the
   * optimality conditions are stated in terms of. */

  Index j = f_N;
  double M = Inf< double >() , best = 0;

  const double * Ki = ( i < f_N ) ? f_K + std::size_t( f_di[ i ] ) * f_n
                                  : nullptr;
  const double Qii = Ki ? v_QD[ i ] : 0;

  for( Index k = 0 ; k < f_N ; ++k ) {
   const double sk = f_ds[ k ];
   const double gk = - sk * v_G[ k ];
   const bool low = ( sk > 0 ) ? ( v_alpha[ k ] > 0 ) : ( v_alpha[ k ] < f_u );

   if( ! low )
    continue;

   if( gk < M )
    M = gk;

   if( ( ! Ki ) || ( gk >= m ) )   // not a violating pair with i
    continue;

   /* The curvature along the direction that moves the two multipliers is
    * K_ii + K_kk - 2 K_ik over the weight of the regularisation term, plus
    * twice the diagonal of the squared loss: the signs cancel out, the
    * direction being the one that keeps s^T alpha where it is. A nonpositive
    * value means that the kernel does not obey Mercer's condition, in which
    * case a tiny curvature is substituted, which sends the step to the
    * farthest reachable end of the segment. */

   double a = Qii + v_QD[ k ] - 2 * ( Ki[ f_di[ k ] ] + f_rb ) / f_rw;
   if( a <= 0 )
    a = dTau;

   const double b = m - gk;
   const double dec = ( b * b ) / a;

   if( dec > best ) { best = dec; j = k; }
   }

  // at optimality the two sides coincide with the multiplier of the equality
  // constraint, i.e., with the bias; in general it is any value in between
  if( ( i < f_N ) && ( M < Inf< double >() ) )
   f_b = ( m + M ) / 2;

  if( ( i == f_N ) || ( j == f_N ) || ( m - M <= f_tol ) )
   return( kOK );  // the optimality conditions hold

  // minimize along the only feasible direction changing alpha_i, alpha_j - -

  const double si = f_ds[ i ] , sj = f_ds[ j ];

  double a = Qii + v_QD[ j ] - 2 * ( Ki[ f_di[ j ] ] + f_rb ) / f_rw;
  if( a <= 0 )
   a = dTau;

  double t = ( m + sj * v_G[ j ] ) / a;  // ( g_i - g_j ) / a

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

  const double * Kj = f_K + std::size_t( f_di[ j ] ) * f_n;

  /* The Hessian is the reweighted Gram matrix divided by the weight of the
   * regularisation term [see Q()], which is read here directly rather than
   * through it: this is the innermost loop of the algorithm, and it only
   * runs over the active set, the gradient of what is out being recomputed
   * when it comes back in. */
  const double di = si * dai / f_rw , dj = sj * daj / f_rw;

  for( Index k = 0 ; k < f_N ; ++k )
   v_G[ k ] += f_ds[ k ] * ( ( Ki[ f_di[ k ] ] + f_rb ) * di +
                             ( Kj[ f_di[ k ] ] + f_rb ) * dj );

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
  * coordinate. Which one is again chosen with second order information, i.e.
  * as the one whose move decreases the objective the most, G_k^2 / Q_kk
  * rather than the largest projected gradient, and the algorithm stops when
  * even the largest violation is small enough. The multipliers that are at a
  * bound and satisfy their own condition are shrunk away exactly as in the
  * other case. */

 while( ( f_max_iter < 0 ) || ( f_iter < Index( f_max_iter ) ) ) {

  Index i = f_N;
  double viol = 0 , best = 0;

  for( Index k = 0 ; k < f_N ; ++k ) {
   const double ak = v_alpha[ k ];
   double pg = v_G[ k ];
   if( ( ak <= 0 ) && ( pg > 0 ) )
    pg = 0;
   else if( ( ak >= f_u ) && ( pg < 0 ) )
    pg = 0;

   if( ! pg )
    continue;

   viol = std::max( viol , std::abs( pg ) );

   double a = v_QD[ k ];
   if( a <= 0 )
    a = dTau;

   const double dec = ( pg * pg ) / a;
   if( dec > best ) { best = dec; i = k; }
   }

  if( ( i == f_N ) || ( viol <= f_tol ) )  // the optimality conditions hold
   return( kOK );

  double a = v_QD[ i ];
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

  const double di = si * dai / f_rw;   // as in solve_with_equality()

  for( Index k = 0 ; k < f_N ; ++k )
   v_G[ k ] += f_ds[ k ] * ( Ki[ f_di[ k ] ] + f_rb ) * di;

  if( f_d )
   v_G[ i ] += f_d * dai;

  ++f_iter;
  }

 return( kStopIter );

 }  // end( SMOSolver::solve_box )

/*--------------------------------------------------------------------------*/
/*------------------------- End File SMOSolver.cpp -------------------------*/
/*--------------------------------------------------------------------------*/
