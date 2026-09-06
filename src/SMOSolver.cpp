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

 const bool was_optimal = f_optimal;

 f_solved = false;
 f_optimal = false;
 f_iter = 0;

 /* If the only thing that has changed is that samples have been *added*, and
  * what was there was optimal, the new multipliers are learnt one at a time
  * along the exact solution path: each of them is grown from zero keeping
  * every other index at its own condition, so that when the last one is done
  * the solution is the optimal one of the new data set and there is nothing
  * left to iterate on [see follow_path()]. The conditions are checked all
  * the same, and the iteration finishes the job if the path has not: an
  * exact statement about a walk of floating point numbers is a statement
  * about what it was meant to do, not about what it did. */

 if( f_path && was_optimal && ( ! f_rmvd ) && ( ! v_new.empty() ) ) {
  bool done = true;
  for( auto c : v_new )
   if( follow_path( c , f_u ) != kOK ) {
    done = false;
    break;
    }

  v_new.clear();

  if( done ) {
   double m = - Inf< double >() , M = Inf< double >();
   for( Index k = 0 ; k < f_N ; ++k ) {
    const double sk = f_ds[ k ] , gk = - sk * v_G[ k ];
    if( ( sk > 0 ) ? ( v_alpha[ k ] < f_u ) : ( v_alpha[ k ] > 0 ) )
     m = std::max( m , gk );
    if( ( sk > 0 ) ? ( v_alpha[ k ] > 0 ) : ( v_alpha[ k ] < f_u ) )
     M = std::min( M , gk );
    }

   if( m - M <= f_tol ) {   // the path has done the whole job
    double v = 0;
    for( Index k = 0 ; k < f_N ; ++k )
     v += v_alpha[ k ] * ( v_G[ k ] + f_dq[ k ] );

    f_value = - v / 2 + f_dc;
    f_solved = true;
    f_optimal = true;

    unlock();

    return( kOK );
    }
   }
  }

 v_new.clear();
 f_rmvd = false;

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
 f_optimal = ( status == kOK );

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
  f_rmvd = true;   // what is gone is gone: the previous solution is no longer
  }                // optimal, whence the path cannot be walked from it

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

 v_s = f_SVM->get_dual_signs();
 v_di_c = f_SVM->get_dual_samples();

 /* The sample of each dual index is read out of the *copy*, and not out of
  * the SVMBlock, because the shrinking reorders the dual indices [see
  * swap_index()]; the copy is put back in the order of the SVMBlock before
  * compute() returns. */
 f_di = v_di_c.data();

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
 v_new.clear();
 f_rmvd = false;

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
 Subset newidx;

 for( Index k = 0 ; k < N ; ++k ) {
  const Index i = v_smap[ di[ k ] ];
  if( i == Inf< Index >() ) {  // a new sample: its multiplier starts at zero
   newidx.push_back( k );
   continue;
   }

  auto it = o_k.find( { i , s[ k ] > 0 } );
  if( it != o_k.end() )
   n_alpha[ k ] = v_alpha[ it->second ];
  else
   newidx.push_back( k );      // a dual index that was not there before
  }

 // the data of the dual, which is the new one from here on
 f_n = f_SVM->get_NSamples();
 f_N = N;
 f_u = f_SVM->get_ub();
 f_d = f_SVM->get_squared_loss() ? 1 / ( 2 * f_SVM->get_C() ) : 0;
 f_rw = f_SVM->get_reg_weight();
 f_K = f_SVM->get_K().data();
 v_s = s;
 v_q = q;
 v_di_c = di;
 f_ds = v_s.data();
 f_dq = v_q.data();
 f_di = v_di_c.data();
 v_alpha = std::move( n_alpha );
 v_new = std::move( newidx );
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

 f_K = f_SVM->get_K().data();   // it may have been moved elsewhere in the
 v_di_c = f_SVM->get_dual_samples();                  // meantime
 f_di = v_di_c.data();

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

/*--------------------------------------------------------------------------*/
/*--------------------- THE EXACT SOLUTION PATH ----------------------------*/
/*--------------------------------------------------------------------------*/

/* The optimality conditions of the dual, written in the form the path walks
 * them, read: with h_k = G_k + s_k b, where b is the multiplier of the
 * equality constraint,
 *
 *   alpha_k = 0      ==>  h_k >= 0
 *   0 < alpha_k < u  ==>  h_k =  0        ( a *free* index, on the margin )
 *   alpha_k = u      ==>  h_k <= 0
 *
 * which is the same thing the SMO iteration checks as
 * max { g_i : i in I_up } <= min { g_j : j in I_low } with g_k = - s_k G_k
 * and b the common value at optimality. */

bool SMOSolver::solve_free_system( const Subset & S , Index c ,
                                   doubleVec & beta , double & beta_b ) const
{
 const Index ns = S.size();
 const Index nb = f_rb ? 0 : 1;   // the border row, if there is an equality
 const Index dim = ns + nb;

 beta.assign( ns , 0 );
 beta_b = 0;

 if( ! dim )   // nothing moves with c, and nothing has to
  return( true );

 std::vector< double > A( dim * dim , 0 ) , r( dim );

 for( Index i = 0 ; i < ns ; ++i ) {
  for( Index j = 0 ; j < ns ; ++j )
   A[ i * dim + j ] = Q( S[ i ] , S[ j ] );

  if( nb )
   A[ i * dim + ns ] = f_ds[ S[ i ] ];

  r[ i ] = - Q( S[ i ] , c );
  }

 if( nb ) {
  for( Index j = 0 ; j < ns ; ++j )
   A[ ns * dim + j ] = f_ds[ S[ j ] ];

  r[ ns ] = - f_ds[ c ];
  }

 // Gaussian elimination with partial pivoting- - - - - - - - - - - - - - - -

 for( Index k = 0 ; k < dim ; ++k ) {
  Index p = k;
  for( Index i = k + 1 ; i < dim ; ++i )
   if( std::abs( A[ i * dim + k ] ) > std::abs( A[ p * dim + k ] ) )
    p = i;

  if( std::abs( A[ p * dim + k ] ) < 1e-12 )
   return( false );   // singular: the path cannot be followed from here

  if( p != k ) {
   for( Index j = k ; j < dim ; ++j )
    std::swap( A[ p * dim + j ] , A[ k * dim + j ] );
   std::swap( r[ p ] , r[ k ] );
   }

  for( Index i = k + 1 ; i < dim ; ++i ) {
   const double f = A[ i * dim + k ] / A[ k * dim + k ];
   if( ! f )
    continue;

   for( Index j = k ; j < dim ; ++j )
    A[ i * dim + j ] -= f * A[ k * dim + j ];

   r[ i ] -= f * r[ k ];
   }
  }

 for( Index k = dim ; k-- > 0 ; ) {
  double v = r[ k ];
  for( Index j = k + 1 ; j < dim ; ++j )
   v -= A[ k * dim + j ] * ( ( j < ns ) ? beta[ j ] : beta_b );

  v /= A[ k * dim + k ];

  if( k < ns )
   beta[ k ] = v;
  else
   beta_b = v;
  }

 return( true );

 }  // end( SMOSolver::solve_free_system )

/*--------------------------------------------------------------------------*/

int SMOSolver::follow_path( Index c , double to )
{
 static const double dPEps = 1e-9;    // what counts as being at a bound
 static const double dPZero = 1e-12;  // what counts as no movement at all

 const Index maxev = 10 * f_N + 100;  // a path that long is a path gone wrong

 auto hof = [ this ]( Index k ) { return( v_G[ k ] + f_ds[ k ] * f_b ); };

 for( Index ev = 0 ; ev < maxev ; ++ev ) {

  // where the multiplier of c has to go, and whether it is there already- -

  const double dist = to - v_alpha[ c ];
  const double dir = ( dist > 0 ) ? 1 : -1;

  if( std::abs( dist ) <= dPZero )
   return( kOK );                  // it has arrived

  const double hc = hof( c );

  if( ( dir > 0 ) && ( hc >= - dPEps ) )
   return( kOK );                  // its own condition holds where it is

  // how the free multipliers and the bias follow it - - - - - - - - - - - -

  /* The margin: the indices whose multiplier is strictly inside its bounds
   * and *also* those that sit at a bound with their own condition holding
   * with equality, which is where a previous event has left them. The latter
   * are on the margin exactly as the former, they simply cannot leave their
   * bound on the wrong side, which is what the pruning below sees to. */

  std::vector< bool > in_S( f_N , false );
  Subset S;

  for( Index k = 0 ; k < f_N ; ++k ) {
   if( k == c )
    continue;

   if( ( ( v_alpha[ k ] > dPEps ) && ( v_alpha[ k ] < f_u - dPEps ) ) ||
       ( std::abs( hof( k ) ) <= dPEps ) ) {
    in_S[ k ] = true;
    S.push_back( k );
    }
   }

  doubleVec beta;
  double beta_b = 0;

  if( S.empty() && ( ! f_rb ) ) {
   /* No multiplier can absorb the movement of c and keep s^T alpha where it
    * is: what moves is the bias alone, until some index stops satisfying its
    * own condition and becomes free, and from there the walk resumes. */

   const double bdir = ( hc < 0 ) ? f_ds[ c ] : - f_ds[ c ];
   double mag = Inf< double >();
   Index who = f_N;

   for( Index k = 0 ; k < f_N ; ++k ) {
    if( k == c )
     continue;

    const double dh = f_ds[ k ] * bdir;   // how h_k moves per unit of bias
    if( std::abs( dh ) <= dPZero )
     continue;

    const double t = - hof( k ) / dh;
    if( ( t > dPZero ) && ( t < mag ) ) { mag = t; who = k; }
    }

   // the bias only has to move until c is happy, if that comes first
   const double tc = - hc / ( f_ds[ c ] * bdir );
   if( ( tc > 0 ) && ( tc <= mag ) ) {
    f_b += bdir * tc;
    return( kOK );
    }

   if( who == f_N )   // nothing stops the bias: the dual is unbounded
    return( kError );

   f_b += bdir * mag;
   continue;
   }

  /* Whoever is at a bound and would be pushed out of it is not on the margin
   * after all: it leaves and the direction is computed again, which can only
   * happen as many times as there are indices on it. */

  for( ; ; ) {
   if( ! solve_free_system( S , c , beta , beta_b ) )
    return( kError );

   Subset kept;
   kept.reserve( S.size() );

   for( Index i = 0 ; i < S.size() ; ++i ) {
    const Index k = S[ i ];
    const double db = beta[ i ] * dir;

    if( ( ( v_alpha[ k ] <= dPEps ) && ( db < - dPZero ) ) ||
        ( ( v_alpha[ k ] >= f_u - dPEps ) && ( db > dPZero ) ) ) {
     in_S[ k ] = false;
     continue;
     }

    kept.push_back( k );
    }

   if( kept.size() == S.size() )
    break;

   S = std::move( kept );
   }

  /* The direction in which the whole state moves: the multipliers of the
   * free indices by beta, the bias by beta_b and, through them, the gradient
   * of every index by the corresponding column of the Hessian. */

  doubleVec w( f_N );
  for( Index k = 0 ; k < f_N ; ++k ) {
   double wk = Q( k , c );
   for( Index i = 0 ; i < S.size() ; ++i )
    wk += Q( k , S[ i ] ) * beta[ i ];
   w[ k ] = wk;
   }

  auto gof = [ & ]( Index k ) {   // how h_k moves per unit of movement of c
   return( w[ k ] + f_ds[ k ] * beta_b );
   };

  // the first event along it- - - - - - - - - - - - - - - - - - - - - - - -

  double mag = std::abs( dist );   // the multiplier of c reaching its target
  int what = 0;
  Index who = f_N;

  { const double gc = gof( c ) * dir;   // c satisfying its own condition
    if( ( dir > 0 ) && ( gc > dPZero ) ) {
     const double t = - hc / gc;
     if( ( t >= 0 ) && ( t < mag ) ) { mag = t; what = 1; }
     }
    }

  for( Index i = 0 ; i < S.size() ; ++i ) {   // a free multiplier bound
   const double db = beta[ i ] * dir;
   if( std::abs( db ) <= dPZero )
    continue;

   const Index k = S[ i ];
   const double room = ( db > 0 ) ? ( f_u - v_alpha[ k ] ) : v_alpha[ k ];
   const double t = room / std::abs( db );
   if( t < mag ) { mag = t; what = 2; who = k; }
   }

  for( Index k = 0 ; k < f_N ; ++k ) {   // an index reaching the margin
   if( ( k == c ) || in_S[ k ] )
    continue;

   const double dh = gof( k ) * dir;
   if( std::abs( dh ) <= dPZero )
    continue;

   const double t = - hof( k ) / dh;
   if( ( t > dPZero ) && ( t < mag ) ) { mag = t; what = 3; who = k; }
   }

  if( ! ( mag < Inf< double >() ) )   // nothing stops the walk
   return( kError );

  // walk that far - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  const double da = dir * mag;

  v_alpha[ c ] += da;
  for( Index i = 0 ; i < S.size() ; ++i )
   v_alpha[ S[ i ] ] += beta[ i ] * da;

  f_b += beta_b * da;

  for( Index k = 0 ; k < f_N ; ++k )
   v_G[ k ] += w[ k ] * da;

  // snap to the bounds, so that the free set is what it looks like- - - - -

  auto snap = [ this ]( Index k ) {
   if( v_alpha[ k ] < dPEps )
    v_alpha[ k ] = 0;
   else
    if( v_alpha[ k ] > f_u - dPEps )
     v_alpha[ k ] = f_u;
   };

  snap( c );
  for( auto k : S )
   snap( k );

  if( what == 2 )
   snap( who );

  ++f_iter;

  if( what == 1 )   // c is on the margin: nothing more is asked of it
   return( kOK );
  }

 return( kError );   // the path did not end where it should have

 }  // end( SMOSolver::follow_path )

/*--------------------------------------------------------------------------*/

int SMOSolver::unlearn( Index i )
{
 if( ! f_SVM )
  throw( std::logic_error( "SMOSolver::unlearn: no SVMBlock is set" ) );

 if( ! f_solved )
  throw( std::logic_error( "SMOSolver::unlearn: there is no solution to "
                           "unlearn a sample from" ) );

 if( i >= f_n )
  throw( std::invalid_argument( "SMOSolver::unlearn: no such sample" ) );

 lock();

 f_iter = 0;
 int status = kOK;

 /* A sample has one multiplier in a classification problem and two in a
  * regression one, and at most one of the two is nonzero: each of them is
  * driven to zero in turn, the ones that are there already costing nothing.
  */

 for( Index k = 0 ; ( k < f_N ) && ( status == kOK ) ; ++k )
  if( f_di[ k ] == i )
   status = follow_path( k , 0 );

 if( status == kOK ) {
  // the value of the dual at the solution, as compute() computes it
  double v = 0;
  for( Index k = 0 ; k < f_N ; ++k )
   v += v_alpha[ k ] * ( v_G[ k ] + f_dq[ k ] );

  f_value = - v / 2 + f_dc;
  }
 else
  f_solved = false;

 unlock();

 return( status );

 }  // end( SMOSolver::unlearn )

/*--------------------------------------------------------------------------*/

void SMOSolver::swap_index( Index a , Index b )
{
 if( a == b )
  return;

 std::swap( v_alpha[ a ] , v_alpha[ b ] );
 std::swap( v_G[ a ] , v_G[ b ] );
 std::swap( v_QD[ a ] , v_QD[ b ] );
 std::swap( v_s[ a ] , v_s[ b ] );
 std::swap( v_q[ a ] , v_q[ b ] );
 std::swap( v_di_c[ a ] , v_di_c[ b ] );
 std::swap( v_perm[ a ] , v_perm[ b ] );

 }  // end( SMOSolver::swap_index )

/*--------------------------------------------------------------------------*/

void SMOSolver::shrink( double m , double M )
{
 for( Index k = 0 ; k < f_act ; ) {
  const double sk = f_ds[ k ];
  const double gk = - sk * v_G[ k ];
  const bool up = ( sk > 0 ) ? ( v_alpha[ k ] < f_u ) : ( v_alpha[ k ] > 0 );
  const bool low = ( sk > 0 ) ? ( v_alpha[ k ] > 0 ) : ( v_alpha[ k ] < f_u );

  /* A multiplier that cannot be increased is of no use if its own value of
   * the bias is larger than the largest one the others allow, since it can
   * then be neither the index the pair is selected from nor the one that
   * makes a violating pair with it, and symmetrically for one that cannot be
   * decreased. "Of no use" is meant at the current multipliers, which is why
   * everything is put back before the conditions are declared to hold. */

  if( ( ( ! up ) && ( gk > m ) ) || ( ( ! low ) && ( gk < M ) ) ) {
   swap_index( k , --f_act );
   continue;   // whatever has just been moved to k has not been looked at
   }

  ++k;
  }
 }  // end( SMOSolver::shrink )

/*--------------------------------------------------------------------------*/

void SMOSolver::shrink_box( double viol )
{
 for( Index k = 0 ; k < f_act ; ) {
  const double ak = v_alpha[ k ];
  const double gk = v_G[ k ];

  // at a bound, and satisfying its own condition by more than what is left
  // to gain elsewhere: a step of the size the others allow cannot wake it up
  if( ( ( ( ak <= 0 ) && ( gk > 0 ) ) || ( ( ak >= f_u ) && ( gk < 0 ) ) ) &&
      ( std::abs( gk ) > viol ) ) {
   swap_index( k , --f_act );
   continue;
   }

  ++k;
  }
 }  // end( SMOSolver::shrink_box )

/*--------------------------------------------------------------------------*/

void SMOSolver::unshrink( void )
{
 if( f_act >= f_N )
  return;

 /* The gradient of what has been left out is stale, the steps taken in the
  * meantime having skipped it: G = Q alpha + q is recomputed here, one row
  * of the Gram matrix per *nonzero* multiplier rather than one per index
  * restored, since a multiplier that is zero contributes nothing. */

 for( Index k = f_act ; k < f_N ; ++k )
  v_G[ k ] = f_dq[ k ] + f_d * v_alpha[ k ];

 for( Index l = 0 ; l < f_N ; ++l ) {
  const double al = v_alpha[ l ];
  if( ! al )
   continue;

  const double cl = f_ds[ l ] * al / f_rw;
  const double * Kl = f_K + std::size_t( f_di[ l ] ) * f_n;

  for( Index k = f_act ; k < f_N ; ++k )
   v_G[ k ] += f_ds[ k ] * ( Kl[ f_di[ k ] ] + f_rb ) * cl;
  }

 f_act = f_N;

 }  // end( SMOSolver::unshrink )

/*--------------------------------------------------------------------------*/

void SMOSolver::restore_order( void )
{
 for( Index k = 0 ; k < f_N ; ++k )
  while( v_perm[ k ] != k )
   swap_index( k , v_perm[ k ] );

 }  // end( SMOSolver::restore_order )

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
  * conditions hold on the active set everything is put back [see
  * unshrink()] and they are checked on the whole index space, the algorithm
  * only stopping if they hold there too. Everything is put back every so
  * often anyway, since an active set that has collapsed can grind on a poor
  * pair while the whole index space would offer a far better step. */

 v_perm.resize( f_N );
 std::iota( v_perm.begin() , v_perm.end() , Index( 0 ) );
 f_act = f_N;

 // the bias interval of the previous iteration, which is what the shrinking
 // is decided on; the first one shrinks nothing, there being no interval yet
 double pm = Inf< double >() , pM = - Inf< double >();

 /* Shrinking at every iteration costs more than it saves: the interval
  * [ M , m ] is computed on the active set, so the more is taken out the
  * narrower it gets and the more the next pass takes out, and the pair that
  * is left to select is a poor one, which is paid in iterations. It is
  * therefore done every so often, as in [Chang and Lin, LIBSVM], which keeps
  * the interval that decides it one of a set that is still large. */

 const Index period = std::min( f_N , Index( 1000 ) );
 Index counter = period;

 /* An active set that has collapsed can also *cost* iterations, and many of
  * them: the pair that is left to select is a poor one, and the algorithm
  * grinds on it while the whole index space would offer a far better step.
  * The state is therefore restored every so often even when nothing asks for
  * it, so that the good steps keep being taken and what the shrinking can
  * cost is bounded by how rarely that happens. */

 Index passes = 0;   // shrinking passes since the last full restore

 while( ( f_max_iter < 0 ) || ( f_iter < Index( f_max_iter ) ) ) {

  if( counter )
   --counter;

  if( f_shrink && ( ! counter ) && ( pM > - Inf< double >() ) ) {
   counter = period;

   if( ( f_act < f_N ) && ( ++passes >= f_patience ) ) {
    unshrink();
    passes = 0;
    }
   else
    shrink( pm , pM );
   }

  // select the first index of the pair: the maximal violating one - - - - -

  Index i = f_act;
  double m = - Inf< double >();

  for( Index k = 0 ; k < f_act ; ++k ) {
   const double sk = f_ds[ k ];
   const double gk = - sk * v_G[ k ];
   const bool up = ( sk > 0 ) ? ( v_alpha[ k ] < f_u ) : ( v_alpha[ k ] > 0 );

   if( up && ( gk > m ) ) { m = gk; i = k; }
   }

  /* Select the second one: the index of I_low that gives the largest
   * decrease among those that make a violating pair with the first. The
   * minimal violating index is computed along the way, since it is what the
   * optimality conditions are stated in terms of. */

  Index j = f_act;
  double M = Inf< double >() , best = 0;

  const double * Ki = ( i < f_act ) ? f_K + std::size_t( f_di[ i ] ) * f_n
                                    : nullptr;
  const double Qii = Ki ? v_QD[ i ] : 0;

  for( Index k = 0 ; k < f_act ; ++k ) {
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
  if( ( i < f_act ) && ( M < Inf< double >() ) )
   f_b = ( m + M ) / 2;

  if( ( i == f_act ) || ( j == f_act ) || ( m - M <= f_tol ) ) {
   if( f_act < f_N ) {
    /* The conditions hold on the active set, which says nothing about the
     * rest: everything comes back, the gradient of what was out is made
     * good again and the conditions are checked where they have to hold. If
     * they do not, what is left is the endgame, and it is played on
     * everything. */
    unshrink();
    passes = 0;
    pm = Inf< double >();
    pM = - Inf< double >();
    continue;
    }

   restore_order();
   return( kOK );  // the optimality conditions hold
   }

  pm = m;
  pM = M;

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

  if( t <= 0 ) {   // no progress is possible along this pair
   if( f_act < f_N ) {
    unshrink();
    passes = 0;
    pm = Inf< double >();
    pM = - Inf< double >();
    continue;
    }

   restore_order();
   return( kOK );  // numerically optimal
   }

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

  for( Index k = 0 ; k < f_act ; ++k )
   v_G[ k ] += f_ds[ k ] * ( ( Ki[ f_di[ k ] ] + f_rb ) * di +
                             ( Kj[ f_di[ k ] ] + f_rb ) * dj );

  if( f_d ) {  // the diagonal term is not part of the kernel expansion
   v_G[ i ] += f_d * dai;
   v_G[ j ] += f_d * daj;
   }

  ++f_iter;
  }

 // the iteration limit is not a reason to leave the state half done
 unshrink();
 restore_order();

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

 v_perm.resize( f_N );
 std::iota( v_perm.begin() , v_perm.end() , Index( 0 ) );
 f_act = f_N;

 // the largest violation of the previous iteration, which is what the
 // shrinking is decided on; the first one shrinks nothing
 double pv = - Inf< double >();

 // as in solve_with_equality(), the active set is thinned out every so often
 // rather than at every iteration, and it is restored every so often even
 // when nothing asks for it, so that a collapsed one cannot grind
 const Index period = std::min( f_N , Index( 1000 ) );
 Index counter = period;
 Index passes = 0;

 while( ( f_max_iter < 0 ) || ( f_iter < Index( f_max_iter ) ) ) {

  if( counter )
   --counter;

  if( f_shrink && ( ! counter ) && ( pv > - Inf< double >() ) ) {
   counter = period;

   if( ( f_act < f_N ) && ( ++passes >= f_patience ) ) {
    unshrink();
    passes = 0;
    }
   else
    shrink_box( pv );
   }

  Index i = f_act;
  double viol = 0 , best = 0;

  for( Index k = 0 ; k < f_act ; ++k ) {
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

  if( ( i == f_act ) || ( viol <= f_tol ) ) {
   if( f_act < f_N ) {   // they may only hold on the active set
    unshrink();
    passes = 0;
    pv = - Inf< double >();
    continue;
    }

   restore_order();
   return( kOK );  // the optimality conditions hold
   }

  pv = viol;

  double a = v_QD[ i ];
  if( a <= 0 )
   a = dTau;

  double ai = v_alpha[ i ] - v_G[ i ] / a;
  ai = std::max( double( 0 ) , std::min( ai , f_u ) );

  const double dai = ai - v_alpha[ i ];
  if( dai == 0 ) {   // no progress is possible along this coordinate
   if( f_act < f_N ) {
    unshrink();
    passes = 0;
    pv = - Inf< double >();
    continue;
    }

   restore_order();
   return( kOK );  // numerically optimal
   }

  v_alpha[ i ] = ai;

  const double si = f_ds[ i ];
  const double * Ki = f_K + std::size_t( f_di[ i ] ) * f_n;

  const double di = si * dai / f_rw;   // as in solve_with_equality()

  for( Index k = 0 ; k < f_act ; ++k )
   v_G[ k ] += f_ds[ k ] * ( Ki[ f_di[ k ] ] + f_rb ) * di;

  if( f_d )
   v_G[ i ] += f_d * dai;

  ++f_iter;
  }

 // the iteration limit is not a reason to leave the state half done
 unshrink();
 restore_order();

 return( kStopIter );

 }  // end( SMOSolver::solve_box )

/*--------------------------------------------------------------------------*/
/*------------------------- End File SMOSolver.cpp -------------------------*/
/*--------------------------------------------------------------------------*/
