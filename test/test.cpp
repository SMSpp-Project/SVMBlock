/*--------------------------------------------------------------------------*/
/*------------------------------ File test.cpp -----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Tester for SVMBlock, SVCBlock, SVRBlock and SMOSolver.
 *
 * The tests exercise all the formulations of the training problem on generated
 * data sets, for the classification and the regression variants, for the
 * linear and the nonlinear kernels and for all the combinations of the
 * squared loss and of the regularised bias. The correctness of the solution
 * found by SMOSolver is asserted by strong duality, i.e., by checking that
 * the value of the primal at the model recovered out of the multipliers is
 * the value the Solver reports, which is a joint check of the solver, of the
 * parametric map of the Block and of the recovery of the model; the abstract
 * representation is checked, in turn, against the same value, which is the
 * same number in every formulation since the Wolfe dual is written as the
 * maximisation that strong duality makes equal to the primal. The consensus
 * rewriting is checked both structurally, against what a generic Lagrangian
 * Solver requires, and numerically, by verifying that at the optimum of the
 * monolithic problem the consensus constraints are satisfied and the
 * sub-Block objectives add up to the monolithic value. The Solution saving
 * the trained model is checked by restoring it into a different Block and
 * verifying that the two predict the same, both directly and after a netCDF
 * round trip.
 *
 * \author Donato Meoli \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 *
 * \copyright &copy; by Donato Meoli
 */
/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include "SVCBlock.h"

#include "SVRBlock.h"

#include "SMOSolver.h"

#include "FRealObjective.h"

#include "LinearFunction.h"

#include "DQuadFunction.h"

#include "ColVariableSolution.h"

#include <cmath>

#include <functional>

#include <iostream>

#include <random>

/*--------------------------------------------------------------------------*/
/*-------------------------------- USING -----------------------------------*/
/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

using Index = Block::Index;
using doubleVec = SVMBlock::doubleVec;

/*--------------------------------------------------------------------------*/
/*------------------------------- GLOBALS ----------------------------------*/
/*--------------------------------------------------------------------------*/

static int failed = 0;

/*--------------------------------------------------------------------------*/
/*------------------------------ FUNCTIONS ---------------------------------*/
/*--------------------------------------------------------------------------*/

static void check( bool ok , const std::string & what )
{
 std::cout << ( ok ? "  ok   " : "  FAIL " ) << what << std::endl;
 if( ! ok )
  ++failed;
 }

/*--------------------------------------------------------------------------*/

static void check_close( double a , double b , double tol ,
                         const std::string & what )
{
 const bool ok = std::abs( a - b ) <= tol * ( 1 + std::abs( b ) );
 std::cout << ( ok ? "  ok   " : "  FAIL " ) << what
           << " ( " << a << " vs " << b << " )" << std::endl;
 if( ! ok )
  ++failed;
 }

/*--------------------------------------------------------------------------*/
/// the value of the primal at the model currently stored in the SVMBlock
/** Computes the value of the primal objective at the model defined by the
 * kernel expansion coefficients and the bias currently stored in \p svm, i.e.,
 * 1/2 c^T K c [ + 1/2 b^2 ] + C sum_k xi_k^p with the slacks set to the
 * smallest values that make the model feasible. Everything is expressed
 * through the Gram matrix, so it is well defined for any kernel. */

static double primal_value( const SVMBlock * svm )
{
 auto & c = svm->get_dual_coefficients();
 auto & K = svm->get_K();
 auto & s = svm->get_dual_signs();
 auto & di = svm->get_dual_samples();
 auto & q = svm->get_dual_costs();

 const Index n = svm->get_NSamples();
 const double b = svm->get_b();

 /* The regularisation term is rw/2 ( || w ||^2 [ + b^2 ] ), and || w ||^2 is
  * the kernel expansion of the model against itself. */
 const double rw = svm->get_reg_weight();

 double v = 0;
 doubleVec f( n , b );

 if( svm->has_linear_term() ) {
  /* The linear term of the primal takes the explicit feature map, hence the
   * weights themselves, which the linear kernel it requires does have. */
  auto & lambda = svm->get_linear_term();
  const auto w = svm->get_w();
  const Index m = svm->get_NFeatures();

  for( Index j = 0 ; j < m ; ++j )
   v += w[ j ] * w[ j ];
  v *= rw / 2;

  if( svm->get_reg_bias() )
   v += rw * b * b / 2;

  for( Index j = 0 ; j < m ; ++j )
   v += ( lambda.empty() ? 0 : lambda[ j ] ) * w[ j ];

  v += svm->get_linear_bias() * b;

  for( Index i = 0 ; i < n ; ++i ) {
   const double * xi = svm->get_x( i );
   for( Index j = 0 ; j < m ; ++j )
    f[ i ] += w[ j ] * xi[ j ];
   }
  }
 else {
  for( Index i = 0 ; i < n ; ++i )
   for( Index j = 0 ; j < n ; ++j )
    v += c[ i ] * c[ j ] * K[ std::size_t( i ) * n + j ];
  v *= rw / 2;

  if( svm->get_reg_bias() )
   v += rw * b * b / 2;

  for( Index i = 0 ; i < n ; ++i )
   for( Index j = 0 ; j < n ; ++j )
    f[ i ] += c[ j ] * K[ std::size_t( j ) * n + i ];
  }

 for( Index k = 0 ; k < s.size() ; ++k ) {
  const double xi = std::max( double( 0 ) ,
                              - q[ k ] - s[ k ] * f[ di[ k ] ] );
  v += svm->get_C() * ( svm->get_squared_loss() ? xi * xi : xi );
  }

 return( v );

 }  // end( primal_value )

/*--------------------------------------------------------------------------*/
/// trains \p svm with SMOSolver and returns the value of the training problem

static double train( SVMBlock * svm , double tol = 1e-10 )
{
 auto solver = Solver::new_Solver( "SMOSolver" );
 if( ! solver )
  throw( std::logic_error( "SMOSolver not present in the Solver factory" ) );

 // the strong duality checks compare the primal at the recovered model with
 // the value of the dual, hence they are only as tight as the solution is
 solver->set_par( SMOSolver::dblSMOTol , tol );

 svm->register_Solver( solver );

 const int status = solver->compute();
 if( status != Solver::kOK )
  throw( std::logic_error( "SMOSolver did not converge" ) );

 solver->get_var_solution();
 const double value = solver->get_var_value();

 svm->unregister_Solver( solver );
 delete solver;

 return( value );

 }  // end( train )

/*--------------------------------------------------------------------------*/
/// sets the Variable of the primal of \p svm to the model ( w , b )
/** Sets the "w" and "b" Variable of the primal formulation of \p svm to the
 * given model and the "xi" ones to the smallest values that make it feasible,
 * which is what they are worth at any optimal solution of the primal. */

static void fill_primal( SVMBlock * svm , const doubleVec & w , double b )
{
 auto wv = svm->get_static_variable_v< ColVariable >( "w" );
 auto xv = svm->get_dynamic_variable< ColVariable >( "xi" );
 auto bv = svm->get_static_variable< ColVariable >( "b" );

 const Index m = svm->get_NFeatures();

 for( Index j = 0 ; j < m ; ++j )
  (*wv)[ j ].set_value( w[ j ] );

 bv->set_value( b );

 auto & s = svm->get_dual_signs();
 auto & di = svm->get_dual_samples();
 auto & q = svm->get_dual_costs();
 auto & X = svm->get_X();

 Index k = 0;
 for( auto & xk : *xv ) {
  double f = b;
  for( Index j = 0 ; j < m ; ++j )
   f += w[ j ] * X[ std::size_t( di[ k ] ) * m + j ];
  xk.set_value( std::max( double( 0 ) , - q[ k ] - s[ k ] * f ) );
  ++k;
  }

 }  // end( fill_primal )

/*--------------------------------------------------------------------------*/
/// the value of the Objective of \p blk at the current Variable

static double objective_value( Block * blk )
{
 auto obj = dynamic_cast< FRealObjective * >( blk->get_objective() );
 obj->compute();
 return( obj->value() );

 }  // end( objective_value )

/*--------------------------------------------------------------------------*/
/// checks that the abstract representation agrees with the physical one
/** Generates the abstract representation of the given formulation, sets its
 * Variable to the model currently stored in \p svm and checks that the value
 * of its Objective is the expected one: the value of the dual for the dual
 * formulation, that of the primal for the primal one. */

static void check_abstract( SVMBlock * svm , int form , double expected ,
                            const std::string & what )
{
 SimpleConfiguration< int > cfg( form );

 svm->generate_abstract_variables( & cfg );
 svm->generate_abstract_constraints();
 svm->generate_objective();

 const Index N = svm->get_NDual();

 if( form == SVMBlock::kWolfeDual ) {
  auto av = svm->get_dynamic_variable< ColVariable >( "alpha" );
  auto & alpha = svm->get_alphas();
  Index k = 0;
  for( auto & ak : *av )
   ak.set_value( alpha[ k++ ] );
  }
 else
  fill_primal( svm , svm->get_w() , svm->get_b() );

 check_close( objective_value( svm ) , expected , 1e-6 , what );

 }  // end( check_abstract )

/*--------------------------------------------------------------------------*/
/// generates a linearly separable two-class data set

static void make_svc_data( Index n , Index m , doubleVec & X , doubleVec & y ,
                           unsigned seed )
{
 std::mt19937 rng( seed );
 std::normal_distribution< double > gauss( 0 , 1 );

 X.resize( std::size_t( n ) * m );
 y.resize( n );

 for( Index i = 0 ; i < n ; ++i ) {
  const double lbl = ( i % 2 ) ? 1 : -1;
  y[ i ] = lbl;
  for( Index j = 0 ; j < m ; ++j )
   X[ std::size_t( i ) * m + j ] = gauss( rng ) + ( j ? 0 : 4 * lbl );
  }

 }  // end( make_svc_data )

/*--------------------------------------------------------------------------*/
/// generates a data set out of an affine function plus a small noise

static void make_svr_data( Index n , Index m , doubleVec & X , doubleVec & y ,
                           unsigned seed )
{
 std::mt19937 rng( seed );
 std::normal_distribution< double > gauss( 0 , 1 );
 std::normal_distribution< double > noise( 0 , 0.05 );

 X.resize( std::size_t( n ) * m );
 y.resize( n );

 doubleVec w( m );
 for( Index j = 0 ; j < m ; ++j )
  w[ j ] = gauss( rng );

 for( Index i = 0 ; i < n ; ++i ) {
  double v = 0.5;
  for( Index j = 0 ; j < m ; ++j ) {
   const double xij = gauss( rng );
   X[ std::size_t( i ) * m + j ] = xij;
   v += w[ j ] * xij;
   }
  y[ i ] = v + noise( rng );
  }

 }  // end( make_svr_data )

/*--------------------------------------------------------------------------*/
/// generates the four-quadrant data set, which no hyperplane can separate

static void make_xor_data( doubleVec & X , doubleVec & y , Index rep )
{
 X.clear();
 y.clear();

 const double base[ 4 ][ 2 ] = { { 1 , 1 } , { -1 , -1 } ,
                                 { 1 , -1 } , { -1 , 1 } };
 const double lbl[ 4 ] = { 1 , 1 , -1 , -1 };

 for( Index r = 0 ; r < rep ; ++r )
  for( Index q = 0 ; q < 4 ; ++q ) {
   const double d = 0.1 * r;
   X.push_back( base[ q ][ 0 ] + d * base[ q ][ 1 ] );
   X.push_back( base[ q ][ 1 ] - d * base[ q ][ 0 ] );
   y.push_back( lbl[ q ] );
   }

 }  // end( make_xor_data )

/*--------------------------------------------------------------------------*/
/// fraction of the samples of \p svm that the model classifies correctly

static double accuracy( const SVMBlock * svm )
{
 const Index n = svm->get_NSamples();
 auto & y = svm->get_y();

 Index right = 0;
 for( Index i = 0 ; i < n ; ++i )
  if( svm->predict( svm->get_x( i ) ) == y[ i ] )
   ++right;

 return( double( right ) / n );

 }  // end( accuracy )

/*--------------------------------------------------------------------------*/
/// solves again with an already attached Solver, returning the value

static double resolve( Solver * solver )
{
 const int status = solver->compute();
 if( status != Solver::kOK )
  throw( std::logic_error( "SMOSolver did not converge" ) );

 solver->get_var_solution();
 return( solver->get_var_value() );

 }  // end( resolve )

/*--------------------------------------------------------------------------*/
/// trains a fresh SVMBlock holding the very same training problem as \p svm
/** Trains, from scratch and with a Solver of its own, a new SVMBlock with the
 * same data set and the same hyper-parameters as \p svm: this is the value
 * that whoever re-optimizes \p svm after a change has to agree with. */

static double from_scratch( const SVMBlock * svm )
{
 auto ref = dynamic_cast< SVMBlock * >(
                                  Block::new_Block( svm->classname() ) );

 svm->copy_hyperparameters( ref );
 ref->load( svm->get_NSamples() , svm->get_NFeatures() , svm->get_X() ,
            svm->get_y() );

 // the linear term is no hyper-parameter, but it is part of the problem
 if( svm->has_linear_term() )
  ref->set_linear_term( svm->get_linear_term() , svm->get_linear_bias() );

 const double value = train( ref );

 delete ref;
 return( value );

 }  // end( from_scratch )

/*--------------------------------------------------------------------------*/
/// trains a fresh SVMBlock holding the training problem of \p svm without i
/** Trains, from scratch and with a Solver of its own, a new SVMBlock with the
 * hyper-parameters of \p svm and its data set *minus* the sample \p i: this
 * is the value that unlearning that sample has to give [see
 * SMOSolver::unlearn()]. */

static double without_sample( const SVMBlock * svm , Index i )
{
 auto ref = dynamic_cast< SVMBlock * >(
                                  Block::new_Block( svm->classname() ) );

 svm->copy_hyperparameters( ref );

 const Index n = svm->get_NSamples() , m = svm->get_NFeatures();
 doubleVec X , y;
 X.reserve( std::size_t( n - 1 ) * m );
 y.reserve( n - 1 );

 for( Index k = 0 ; k < n ; ++k ) {
  if( k == i )
   continue;

  y.push_back( svm->get_y()[ k ] );
  for( Index j = 0 ; j < m ; ++j )
   X.push_back( svm->get_X()[ std::size_t( k ) * m + j ] );
  }

 ref->load( n - 1 , m , X , y );

 if( svm->has_linear_term() )
  ref->set_linear_term( svm->get_linear_term() , svm->get_linear_bias() );

 const double value = train( ref );

 delete ref;
 return( value );

 }  // end( without_sample )

/*--------------------------------------------------------------------------*/
/// the sample of \p svm carrying the largest multiplier, and one carrying none
/** Reads the multipliers of the solution \p svm holds and returns, in \p sv,
 * the sample whose multipliers are the largest, i.e. a support vector, and in
 * \p nsv one whose multipliers are all zero, i.e. a sample the model does not
 * lean on; either is left at get_NSamples() if there is none. */

static void find_support( const SVMBlock * svm , Index & sv , Index & nsv )
{
 const Index n = svm->get_NSamples();
 doubleVec mass( n , 0 );

 auto & a = svm->get_alphas();
 auto & di = svm->get_dual_samples();

 for( Index k = 0 ; k < a.size() ; ++k )
  mass[ di[ k ] ] += std::abs( a[ k ] );

 sv = nsv = n;
 double top = 1e-8;

 for( Index i = 0 ; i < n ; ++i )
  if( mass[ i ] > top ) { top = mass[ i ]; sv = i; }
  else
   if( ( mass[ i ] <= 1e-12 ) && ( nsv == n ) )
    nsv = i;

 }  // end( find_support )

/*--------------------------------------------------------------------------*/
/// checks that a change is followed by the abstract representation
/** Builds two SVMBlock of the given kind holding the same data set, and
 * subjects both to \p change: the first one after having generated its
 * abstract representation of the given formulation, so that the latter has to
 * be updated, the second one before, so that it is generated already changed.
 * The two must then encode the same problem, which is checked on the value of
 * the Objective at the optimal model and on the sides of the constraints. */

static void check_change( const std::string & kind , int form ,
                          const doubleVec & X , const doubleVec & y ,
                          Index n , Index m ,
                          const std::function< void( SVMBlock * ) > & change ,
                          const std::string & what )
{
 auto a = dynamic_cast< SVMBlock * >( Block::new_Block( kind ) );
 auto b = dynamic_cast< SVMBlock * >( Block::new_Block( kind ) );

 a->load( n , m , X , y );
 b->load( n , m , X , y );

 SimpleConfiguration< int > cfg( form );

 a->generate_abstract_variables( & cfg );
 a->generate_abstract_constraints();
 a->generate_objective();

 change( a );  // the abstract representation has to follow

 change( b );  // while here it is generated once the change is done
 b->generate_abstract_variables( & cfg );
 b->generate_abstract_constraints();
 b->generate_objective();

 // the optimal model of the changed problem, which both must agree on
 const double value = train( b );
 a->set_dual_solution( doubleVec( b->get_alphas() ) , b->get_b() );

 a->set_solution_in_abstract();
 b->set_solution_in_abstract();

 check_close( objective_value( a ) , value , 1e-6 ,
              what + ": the Objective is the optimal value" );
 check_close( objective_value( a ) , objective_value( b ) , 1e-9 ,
              what + ": the same Objective as if generated anew" );

 bool ok = ( a->get_dynamic_constraints().size() ==
             b->get_dynamic_constraints().size() );

 if( form == SVMBlock::kWolfeDual ) {
  auto ba = a->get_dynamic_constraint< LB0Constraint >( "box" );
  auto bb = b->get_dynamic_constraint< LB0Constraint >( "box" );
  ok &= ba && bb && ( ba->size() == bb->size() );
  if( ok ) {
   auto bbk = bb->begin();
   for( auto & bak : *ba )
    ok &= ( bak.get_rhs() == (bbk++)->get_rhs() );
   }
  }
 else {
  auto ca = a->get_dynamic_constraint< FRowConstraint >( "cons" );
  auto cb = b->get_dynamic_constraint< FRowConstraint >( "cons" );
  ok &= ca && cb && ( ca->size() == cb->size() );
  if( ok ) {
   auto cbk = cb->begin();
   for( auto & cak : *ca )
    ok &= ( cak.get_lhs() == (cbk++)->get_lhs() );
   }
  }

 check( ok , what + ": the same Constraint as if generated anew" );

 delete a;
 delete b;

 }  // end( check_change )

/*--------------------------------------------------------------------------*/
/*-------------------------------- main() ----------------------------------*/
/*--------------------------------------------------------------------------*/

int main( int argc , char ** argv )
{
 // the factories - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

 std::cout << "factories" << std::endl;
 {
  auto blk = Block::new_Block( "SVCBlock" );
  check( dynamic_cast< SVCBlock * >( blk ) , "SVCBlock in the Block factory" );
  delete blk;

  blk = Block::new_Block( "SVRBlock" );
  check( dynamic_cast< SVRBlock * >( blk ) , "SVRBlock in the Block factory" );
  delete blk;

  auto slv = Solver::new_Solver( "SMOSolver" );
  check( dynamic_cast< SMOSolver * >( slv ) ,
         "SMOSolver in the Solver factory" );
  delete slv;
  }

 // the linear classifier - - - - - - - - - - - - - - - - - - - - - - - - - -

 std::cout << "linear SVC" << std::endl;
 {
  doubleVec X , y;
  make_svc_data( 60 , 3 , X , y , 1 );

  for( int sl = 0 ; sl < 2 ; ++sl )
   for( int rb = 0 ; rb < 2 ; ++rb ) {
    SVCBlock svm;
    svm.set_kernel( SVMBlock::kLinear );
    svm.set_C( 10 );
    svm.set_squared_loss( sl );
    svm.set_reg_bias( rb );
    svm.load( 60 , 3 , X , y );

    const double value = train( & svm );
    const std::string tag = std::string( sl ? "L2" : "L1" ) +
                            ( rb ? " reg bias" : " free bias" );

    check_close( primal_value( & svm ) , value , 1e-4 ,
                 "strong duality, " + tag );
    check( accuracy( & svm ) == 1 , "separates the data, " + tag );
    check_close( svm.dual_objective( svm.get_alphas() ) , value , 1e-8 ,
                 "dual objective, " + tag );

    check_abstract( & svm , SVMBlock::kWolfeDual , value ,
                    "abstract dual, " + tag );
    }
  }

 /* The weight of the regularisation term, which is what a chunk of the
  * consensus structure carries a share of: it divides the Hessian of the
  * dual, so this is the check that a Solver reading the dual, and the model
  * it recovers from the multipliers, follow it. */

 std::cout << "the weight of the regularisation term" << std::endl;
 {
  doubleVec X , y;
  make_svc_data( 40 , 3 , X , y , 3 );

  for( double rw : { 1.0 , 0.5 , 0.25 } ) {
   const std::string tag = "rw = " + std::to_string( rw );

   SVCBlock svm;
   svm.set_kernel( SVMBlock::kLinear );
   svm.set_C( 1 );
   svm.set_reg_weight( rw );
   svm.load( 40 , 3 , X , y );

   const double value = train( & svm );

   check_close( primal_value( & svm ) , value , 1e-6 ,
                "strong duality, " + tag );
   check_close( svm.dual_objective( svm.get_alphas() ) , value , 1e-8 ,
                "dual objective, " + tag );
   check_abstract( & svm , SVMBlock::kWolfeDual , value ,
                   "abstract dual, " + tag );

   SVCBlock prm;
   prm.set_kernel( SVMBlock::kLinear );
   prm.set_C( 1 );
   prm.set_reg_weight( rw );
   prm.load( 40 , 3 , X , y );
   train( & prm );   // check_abstract() fills the Variable with the model
   check_abstract( & prm , SVMBlock::kPrimal , value ,
                   "abstract primal, " + tag );
   }

  // and it can be changed with the abstract representation there
  for( int form = SVMBlock::kWolfeDual ; form <= SVMBlock::kPrimal ; ++form )
   check_change( "SVCBlock" , form , X , y , 40 , 3 ,
                 []( SVMBlock * svm ) { svm->set_reg_weight( 0.4 ); } ,
                 std::string( form == SVMBlock::kWolfeDual ? "dual: "
                                                           : "primal: " ) +
                 "the weight of the regularisation" );
  }

 // the linear term of the primal - - - - - - - - - - - - - - - - - - - - - -
 //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 /* What the Lagrangian relaxation of the constraints linking a chunk of the
  * consensus structure to the others leaves in the subproblem of that chunk:
  * that the primal at the recovered model and the dual at the multipliers
  * agree is what certifies that both are optimal. */

 std::cout << "the linear term of the primal" << std::endl;
 {
  doubleVec X , y;
  make_svc_data( 40 , 3 , X , y , 11 );

  doubleVec lambda = { 0.7 , -0.4 , 0.2 };

  for( int rb = 0 ; rb < 2 ; ++rb )
   for( double rw : { 1.0 , 0.5 } )
    for( double mu : { 0.0 , 0.3 , -1.5 } ) {
     const std::string tag = std::string( rb ? "reg. bias, " : "" ) +
                             "rw = " + std::to_string( rw ) +
                             ", mu = " + std::to_string( mu );

     SVCBlock svm;
     svm.set_kernel( SVMBlock::kLinear );
     svm.set_C( 1 );
     svm.set_reg_bias( rb );
     svm.set_reg_weight( rw );
     svm.load( 40 , 3 , X , y );
     svm.set_linear_term( lambda , mu );

     const double value = train( & svm );

     check_close( primal_value( & svm ) , value , 1e-6 ,
                  "strong duality, " + tag );
     check_close( svm.dual_objective( svm.get_alphas() ) , value , 1e-8 ,
                  "dual objective, " + tag );
     check_abstract( & svm , SVMBlock::kWolfeDual , value ,
                     "abstract dual, " + tag );

     // the equality constraint of the dual has mu as its right-hand side
     if( ! rb ) {
      auto eq = svm.get_static_constraint< FRowConstraint >( "eq" );
      check( eq && ( eq->get_rhs() == mu ) && ( eq->get_lhs() == mu ) ,
             "the right-hand side of the equality, " + tag );
      }

     SVCBlock prm;
     prm.set_kernel( SVMBlock::kLinear );
     prm.set_C( 1 );
     prm.set_reg_bias( rb );
     prm.set_reg_weight( rw );
     prm.load( 40 , 3 , X , y );
     prm.set_linear_term( lambda , mu );
     train( & prm );  // check_abstract() fills the Variable with the model
     check_abstract( & prm , SVMBlock::kPrimal , value ,
                     "abstract primal, " + tag );
     }

  /* A Lagrangian Solver changes the multipliers at each of its iterations,
   * and it does so by writing them into the Objective of the chunk: that
   * this is mapped back into the physical representation is what lets the
   * Solver reading the latter be used inside such a scheme, and each of the
   * subproblems is re-optimized from the previous one. */
  { SVCBlock svm;
    svm.set_kernel( SVMBlock::kLinear );
    svm.set_C( 1 );
    svm.load( 40 , 3 , X , y );

    SimpleConfiguration< int > cfg( SVMBlock::kPrimal );
    svm.generate_abstract_variables( & cfg );
    svm.generate_abstract_constraints();
    svm.generate_objective();

    auto solver = Solver::new_Solver( "SMOSolver" );
    solver->set_par( SMOSolver::dblSMOTol , 1e-10 );
    svm.register_Solver( solver );
    resolve( solver );

    auto qf = static_cast< DQuadFunction * >(
     static_cast< FRealObjective * >( svm.get_objective() )->get_function() );

    for( double t : { 1.0 , -0.5 , 2.0 , 0.0 } ) {
     const std::string tag = "t = " + std::to_string( t );

     // what a Lagrangian Solver writes into the Objective of the chunk
     doubleVec lin( 4 );
     for( Index j = 0 ; j < 3 ; ++j )
      lin[ j ] = t * lambda[ j ];
     lin[ 3 ] = t * 0.2;

     qf->modify_linear_coefficients( doubleVec( lin ) ,
                                     Block::Subset( { 0 , 1 , 2 , 3 } ) ,
                                     true ,
                                     eModBlck );

     bool ok = ( svm.get_linear_bias() == lin[ 3 ] );
     if( t )
      ok &= ( svm.get_linear_term() == doubleVec( lin.begin() ,
                                                  lin.begin() + 3 ) );
     else
      ok &= ( ! svm.has_linear_term() );

     check( ok , "the linear term is mapped back, " + tag );

     check_close( resolve( solver ) , from_scratch( & svm ) , 1e-6 ,
                  "re-optimization after the linear term, " + tag );
     }

    svm.unregister_Solver( solver );
    delete solver;
    }

  /* The feasible set of the dual is { 0 <= alpha <= C , s^T alpha = mu },
   * which is empty as soon as mu goes beyond C times the number of dual
   * indices of the corresponding sign: the subproblem of the chunk is then
   * unbounded in the bias, and this has to be *proved*, not guessed, since a
   * wrong answer here would silently become a wrong Lagrangian bound. */
  { const Index np = 20;   // the samples of each class, hence C np = 20

    for( double mu : { 19.9 , 20.0 , 20.1 , -20.0 , -20.1 } ) {
     SVCBlock svm;
     svm.set_kernel( SVMBlock::kLinear );
     svm.set_C( 1 );
     svm.load( 40 , 3 , X , y );
     svm.set_linear_term( lambda , mu );

     auto solver = Solver::new_Solver( "SMOSolver" );
     solver->set_par( SMOSolver::dblSMOTol , 1e-10 );
     svm.register_Solver( solver );

     const int status = solver->compute();
     const bool empty = std::abs( mu ) > np;
     const std::string tag = "mu = " + std::to_string( mu );

     if( empty )
      check( status == Solver::kUnbounded ,
             "the empty dual is proved empty, " + tag );
     else {
      solver->get_var_solution();
      double sa = 0;
      auto & alpha = svm.get_alphas();
      auto & sg = svm.get_dual_signs();
      for( Index k = 0 ; k < alpha.size() ; ++k )
       sa += sg[ k ] * alpha[ k ];

      check( ( status == Solver::kOK ) && ( std::abs( sa - mu ) < 1e-9 ) ,
             "the equality constraint is satisfied, " + tag );
      }

     svm.unregister_Solver( solver );
     delete solver;
     }
    }

  // a zero linear term is the same as no linear term at all
  { SVCBlock svm;
    svm.set_kernel( SVMBlock::kLinear );
    svm.set_C( 1 );
    svm.load( 40 , 3 , X , y );
    const double value = train( & svm );

    svm.set_linear_term( doubleVec( 3 , 0 ) , 0 );
    check( ! svm.has_linear_term() , "a zero linear term is no linear term" );
    check_close( train( & svm ) , value , 1e-9 , "the same optimal value" );
    }

  // and it can be changed with the abstract representation there
  for( int form = SVMBlock::kWolfeDual ; form <= SVMBlock::kPrimal ; ++form )
   check_change( "SVCBlock" , form , X , y , 40 , 3 ,
                 [ &lambda ]( SVMBlock * svm ) {
                  svm->set_kernel( SVMBlock::kLinear );
                  svm->set_linear_term( lambda , 0.2 );
                  } ,
                 std::string( form == SVMBlock::kWolfeDual ? "dual: "
                                                           : "primal: " ) +
                 "the linear term" );
  }

 // the same, but through the primal formulation- - - - - - - - - - - - - - -

 std::cout << "primal formulation" << std::endl;
 {
  doubleVec X , y;
  make_svc_data( 40 , 2 , X , y , 7 );

  for( int sl = 0 ; sl < 2 ; ++sl ) {
   SVCBlock svm;
   svm.set_kernel( SVMBlock::kLinear );
   svm.set_C( 5 );
   svm.set_squared_loss( sl );
   svm.load( 40 , 2 , X , y );

   const double value = train( & svm );
   const std::string tag = std::string( sl ? "L2" : "L1" );

   check_abstract( & svm , SVMBlock::kPrimal , value ,
                   "abstract primal, " + tag );

   // the value does not depend on the formulation, and the Solver leaves the
   // model in the Variable, whence the Objective evaluates to that very same
   // value without anybody filling them by hand
   check_close( train( & svm ) , value , 1e-8 ,
                "the value does not depend on the formulation, " + tag );
   check_close( objective_value( & svm ) , value , 1e-6 ,
                "SMOSolver writes the model in the primal Variable, " + tag );
   }
  }

 // the nonlinear classifier- - - - - - - - - - - - - - - - - - - - - - - - -

 std::cout << "nonlinear SVC" << std::endl;
 {
  doubleVec X , y;
  make_xor_data( X , y , 5 );
  const Index n = y.size();

  {
   SVCBlock svm;
   svm.set_kernel( SVMBlock::kLinear );
   svm.set_C( 10 );
   svm.load( n , 2 , X , y );
   train( & svm );
   check( accuracy( & svm ) < 1 ,
          "the linear kernel cannot separate the four quadrants" );
   }

  for( int krn = SVMBlock::kPoly ; krn <= SVMBlock::kLaplacian ; ++krn ) {
   SVCBlock svm;
   svm.set_kernel( krn , SVMBlock::dGammaScale , 2 , 1 );
   svm.set_C( 100 );
   svm.load( n , 2 , X , y );

   const double value = train( & svm );
   const std::string tag = "kernel " + std::to_string( krn );

   check_close( primal_value( & svm ) , value , 1e-4 ,
                "strong duality, " + tag );
   check( accuracy( & svm ) == 1 , "separates the data, " + tag );
   }
  }

 // the regression- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

 std::cout << "SVR" << std::endl;
 {
  doubleVec X , y;
  make_svr_data( 50 , 3 , X , y , 3 );

  for( int sl = 0 ; sl < 2 ; ++sl )
   for( int rb = 0 ; rb < 2 ; ++rb ) {
    SVRBlock svm;
    svm.set_kernel( SVMBlock::kLinear );
    svm.set_C( 100 );
    svm.set_epsilon( 0.1 );
    svm.set_squared_loss( sl );
    svm.set_reg_bias( rb );
    svm.load( 50 , 3 , X , y );

    const double value = train( & svm );
    const std::string tag = std::string( sl ? "L2" : "L1" ) +
                            ( rb ? " reg bias" : " free bias" );

    check_close( primal_value( & svm ) , value , 1e-4 ,
                 "strong duality, " + tag );
    check_abstract( & svm , SVMBlock::kWolfeDual , value ,
                    "abstract dual, " + tag );

    double worst = 0;
    for( Index i = 0 ; i < 50 ; ++i )
     worst = std::max( worst ,
                       std::abs( svm.predict( svm.get_x( i ) ) - y[ i ] ) );
    check( worst < 0.5 , "fits the data, " + tag );
    }
  }

 // the nonlinear regression- - - - - - - - - - - - - - - - - - - - - - - - -

 std::cout << "nonlinear SVR" << std::endl;
 {
  doubleVec X( 40 ) , y( 40 );
  for( Index i = 0 ; i < 40 ; ++i ) {
   X[ i ] = -2 + 4 * double( i ) / 39;
   y[ i ] = std::sin( 3 * X[ i ] );
   }

  SVRBlock svm;
  svm.set_kernel( SVMBlock::kGaussian , 1 );
  svm.set_C( 100 );
  svm.set_epsilon( 0.01 );
  svm.load( 40 , 1 , X , y );

  const double value = train( & svm );
  check_close( primal_value( & svm ) , value , 1e-4 ,
               "strong duality, gaussian SVR" );

  double worst = 0;
  for( Index i = 0 ; i < 40 ; ++i )
   worst = std::max( worst ,
                     std::abs( svm.predict( svm.get_x( i ) ) - y[ i ] ) );
  check( worst < 0.1 , "fits a sine, gaussian SVR" );
  }

 // the consensus rewriting - - - - - - - - - - - - - - - - - - - - - - - - -

 std::cout << "consensus rewriting" << std::endl;
 {
  const Index n = 48 , m = 3 , P = 4;

  doubleVec X , y;
  make_svc_data( n , m , X , y , 5 );

  // the reference: the same problem written as one Block
  SVCBlock ref;
  ref.set_kernel( SVMBlock::kLinear );
  ref.set_C( 3 );
  ref.load( n , m , X , y );

  const double value = train( & ref );
  const auto w = ref.get_w();
  const double b = ref.get_b();

  /* The rewriting is a *structure* the SVMBlock is given, i.e., the P chunks
   * are its sub-Block, and it is chosen exactly as any other structure is,
   * through set_structure(); the abstract representation follows. */
  SVCBlock cns;
  cns.set_kernel( SVMBlock::kLinear );
  cns.set_C( 3 );
  cns.load( n , m , X , y );

  SimpleConfiguration< int > chunks( P );
  cns.set_structure( & chunks );
  cns.generate_abstract_variables();
  cns.generate_abstract_constraints();
  cns.generate_objective();

  Block * dec = & cns;

  // the structure a generic Lagrangian Solver expects
  check( ! dec->get_static_variable_v< ColVariable >( "w" ) &&
         ! dec->get_dynamic_variable< ColVariable >( "alpha" ) ,
         "no Variable in the father Block" );
  check( dec->get_number_nested_Blocks() == P , "one sub-Block per chunk" );
  check( dec->get_objective() &&
         ( dec->get_objective()->get_num_active_var() == 0 ) ,
         "the Objective of the father Block is empty" );
  check( cns.get_NChunks() == P , "the SVMBlock says how many chunks" );

  auto link = dec->get_static_constraint_v< FRowConstraint >( "link" );
  check( link && ( link->size() == ( P - 1 ) * ( m + 1 ) ) ,
         "( P - 1 )( m + 1 ) consensus constraints" );

  bool all_linear = true;
  for( auto & lnk : *link )
   if( ! dynamic_cast< LinearFunction * >( lnk.get_function() ) )
    all_linear = false;
  check( all_linear , "the consensus constraints are linear" );

  // every sample belongs to exactly one chunk, and every chunk sees both
  // classes, which is what keeps its Lagrangian subproblem bounded
  Index tot = 0;
  bool both = true;
  for( Index p = 0 ; p < P ; ++p ) {
   auto sub = dynamic_cast< SVMBlock * >( dec->get_nested_Block( p ) );
   tot += sub->get_NSamples();
   auto & ys = sub->get_y();
   if( std::count( ys.begin() , ys.end() , 1. ) == 0 ||
       std::count( ys.begin() , ys.end() , -1. ) == 0 )
    both = false;
   }
  check( tot == n , "the chunks partition the samples" );
  check( both , "every chunk sees both classes" );

  // the rewriting is exact: at the optimum of the problem written as one
  // Block the consensus constraints are satisfied and the sub-Block
  // objectives add up to its value
  double sum = 0;
  for( Index p = 0 ; p < P ; ++p ) {
   auto sub = dynamic_cast< SVMBlock * >( dec->get_nested_Block( p ) );
   fill_primal( sub , w , b );
   sum += objective_value( sub );
   }

  double viol = 0;
  for( auto & lnk : *link ) {
   auto f = lnk.get_function();
   f->compute();
   viol = std::max( viol , std::abs( f->get_value() ) );
   }

  check( viol < 1e-12 , "the consensus constraints are satisfied" );
  check_close( sum , value , 1e-8 , "the rewriting is exact" );

  // every copy of the model is the model, so it is read out of any sub-Block
  auto sub = dynamic_cast< SVMBlock * >( dec->get_nested_Block( 0 ) );
  sub->get_solution_from_abstract();
  double dw = std::abs( sub->get_b() - b );
  auto w2 = sub->get_w();
  for( Index j = 0 ; j < m ; ++j )
   dw = std::max( dw , std::abs( w2[ j ] - w[ j ] ) );
  check( dw < 1e-12 , "the model is read out of a sub-Block" );

  /* Each chunk is a SVMBlock of its own, hence it produces its own
   * SVMBlockSolution: this is what a generic Solver working on the primal of
   * a chunk, which knows nothing about SVMBlock, leaves behind. */
  SimpleConfiguration< int > model_cfg( 3 );
  auto csol = dynamic_cast< SVMBlockSolution * >(
				  sub->get_Solution( & model_cfg , false ) );
  double dsol = csol ? std::abs( csol->get_b() - b ) : 1;
  if( csol )
   for( Index j = 0 ; j < m ; ++j )
    dsol = std::max( dsol , std::abs( csol->get_w()[ j ] - w[ j ] ) );
  check( dsol < 1e-12 , "the chunk produces its own SVMBlockSolution" );
  delete csol;

  /* The model of the SVMBlock is that of any of its chunks, the consensus
   * constraints making them the same one: it is therefore read out of the
   * father as it is out of a monolithic one. */
  cns.get_solution_from_abstract();
  double dfather = std::abs( cns.get_b() - b );
  auto wf = cns.get_w();
  for( Index j = 0 ; j < m ; ++j )
   dfather = std::max( dfather , std::abs( wf[ j ] - w[ j ] ) );
  check( dfather < 1e-12 , "the model is read out of the father Block" );

  /* A hyper-parameter can be changed with the structure in place: no local
   * change of the abstract representation can do it, the chunks holding a
   * copy of it, hence they are dealt out anew and the rewriting has to stay
   * exact. */
  ref.set_C( 5 );
  const double nvalue = train( & ref );
  const auto nw = ref.get_w();
  const double nb = ref.get_b();

  cns.set_C( 5 );

  double nsum = 0;
  for( Index p = 0 ; p < P ; ++p ) {
   auto sub = dynamic_cast< SVMBlock * >( dec->get_nested_Block( p ) );
   fill_primal( sub , nw , nb );
   nsum += objective_value( sub );
   }

  check_close( nsum , nvalue , 1e-8 ,
               "the rewriting is exact after a hyper-parameter changes" );

  // the structure can no longer be changed once the abstract representation
  // is there, the Constraint tying the chunks that are there
  bool caught = false;
  try {
   SimpleConfiguration< int > two( 2 );
   cns.set_structure( & two );
   }
  catch( const std::exception & e ) { caught = true; }
  check( caught , "the structure cannot be changed after the AR" );

  // a chunk of a single class would have an unbounded subproblem
  caught = false;
  try {
   SVCBlock bad;
   bad.set_C( 3 );
   bad.load( n , m , X , y );
   SimpleConfiguration< int > all( n );
   bad.set_structure( & all );
   }
  catch( const std::exception & e ) { caught = true; }
  check( caught , "single-class chunks are refused" );

  // and no rewriting at all is possible without an explicit feature map
  caught = false;
  try {
   SVCBlock nl;
   nl.set_kernel( SVMBlock::kGaussian );
   nl.load( n , m , X , y );
   SimpleConfiguration< int > two( 2 );
   nl.set_structure( & two );
   }
  catch( const std::exception & e ) { caught = true; }
  check( caught , "a nonlinear kernel is refused" );
  }

 // the Benders structure - - - - - - - - - - - - - - - - - - - - - - - - - -
 //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 /* The other way of splitting the very same sum along the samples: the model
  * and the regularisation term stay in the SVMBlock, which is the master,
  * and each chunk holds the slacks of its samples and its share of the loss.
  * Projecting the slacks out of a chunk leaves the loss of that chunk at the
  * model, which is what a Benders Solver approximates from below. */

 std::cout << "Benders structure" << std::endl;
 {
  const Index n = 48 , m = 3 , P = 4;

  doubleVec X , y;
  make_svc_data( n , m , X , y , 5 );

  // the reference: the same problem written as one Block
  SVCBlock ref;
  ref.set_kernel( SVMBlock::kLinear );
  ref.set_C( 3 );
  ref.load( n , m , X , y );

  const double value = train( & ref );
  const auto w = ref.get_w();
  const double b = ref.get_b();

  SVCBlock ben;
  ben.set_kernel( SVMBlock::kLinear );
  ben.set_C( 3 );
  ben.load( n , m , X , y );

  // the structure says two things, the decomposition and the chunks
  SimpleConfiguration< std::pair< int , int > > bcfg(
   std::make_pair( int( SVMBlock::kBenders ) , int( P ) ) );
  ben.set_structure( & bcfg );

  ben.generate_abstract_variables();
  ben.generate_abstract_constraints();
  ben.generate_objective();

  check( ben.get_structure_type() == SVMBlock::kBenders ,
         "the SVMBlock says which decomposition it is" );
  check( ben.get_number_nested_Blocks() == P , "one sub-Block per chunk" );

  // the master has the model, and only that
  check( ben.get_static_variable_v< ColVariable >( "w" ) &&
         ben.get_static_variable< ColVariable >( "b" ) &&
         ( ! ben.get_dynamic_variable< ColVariable >( "xi" ) ) &&
         ( ! ben.get_dynamic_variable< ColVariable >( "alpha" ) ) ,
         "the master has the model and no slack" );

  check( ben.get_static_constraints().empty() &&
         ben.get_dynamic_constraints().empty() ,
         "the master has no Constraint, the model being free" );

  // every dual index is in exactly one chunk, with its slack and its margin
  Index tot = 0;
  bool right = true;
  for( Index p = 0 ; p < P ; ++p ) {
   auto sub = ben.get_nested_Block( p );
   auto xi = sub->get_dynamic_variable< ColVariable >( "xi" );
   auto cn = sub->get_dynamic_constraint< FRowConstraint >( "cons" );
   auto xb = sub->get_dynamic_constraint< LB0Constraint >( "xibox" );

   if( ( ! xi ) || ( ! cn ) || ( ! xb ) ||
       ( xi->size() != cn->size() ) || ( xi->size() != xb->size() ) )
    right = false;
   else
    tot += xi->size();
   }

  check( right , "each chunk has its slacks, their bounds and their margins" );
  check( tot == ben.get_NDual() , "the chunks partition the dual indices" );

  /* The decomposition is exact: at the optimum of the problem written as one
   * Block, the Objective of the master, i.e. the regularisation term, plus
   * the Objective of the chunks, i.e. the loss of their samples, add up to
   * its value. */

  ben.set_primal_solution( doubleVec( w ) , b );
  ben.set_solution_in_abstract();

  double sum = objective_value( & ben );
  for( Index p = 0 ; p < P ; ++p )
   sum += objective_value( ben.get_nested_Block( p ) );

  check_close( sum , value , 1e-8 , "the Benders structure is exact" );

  // and the model is read out of the master, which is where it lives
  ben.get_solution_from_abstract();
  double dw = std::abs( ben.get_b() - b );
  auto w2 = ben.get_w();
  for( Index j = 0 ; j < m ; ++j )
   dw = std::max( dw , std::abs( w2[ j ] - w[ j ] ) );
  check( dw < 1e-12 , "the model is read out of the master" );

  // the plain SimpleConfiguration< int > is still the consensus one
  SVCBlock cns;
  cns.set_kernel( SVMBlock::kLinear );
  cns.set_C( 3 );
  cns.load( n , m , X , y );
  SimpleConfiguration< int > four( P );
  cns.set_structure( & four );
  check( ( cns.get_structure_type() == SVMBlock::kConsensus ) &&
         ( cns.get_NChunks() == P ) ,
         "a plain int is the consensus structure" );
  }

 // changing the training problem- - - - - - - - - - - - - - - - - - - - - - -

 std::cout << "re-optimization" << std::endl;
 {
  const Index n = 50 , m = 3;
  doubleVec X , y;
  make_svc_data( n , m , X , y , 13 );

  /* The Solver stays attached across the changes, hence it has to make sense
   * of the Modification the SVMBlock issues: whatever it does, it has to
   * agree with a training done from scratch. */
  SVCBlock svm;
  svm.load( n , m , X , y );

  auto solver = Solver::new_Solver( "SMOSolver" );
  auto smo = dynamic_cast< SMOSolver * >( solver );
  solver->set_par( SMOSolver::dblSMOTol , 1e-10 );
  svm.register_Solver( solver );

  const double value = resolve( solver );
  check_close( value , from_scratch( & svm ) , 1e-8 , "the first training" );

  // nothing has changed, hence there is nothing left to do
  const double again = resolve( solver );
  check( ( again == value ) && ( ! smo->get_iter() ) ,
         "re-solving an unchanged SVMBlock takes no iteration" );

  // the trade-off parameter: the bound on the multipliers, hence a scaling
  svm.set_C( 10 );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "C increased" );

  svm.set_C( 0.05 );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "C decreased" );

  // the shape of the loss: the bound and the diagonal of the Hessian
  svm.set_squared_loss( true );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "squared loss" );

  svm.set_C( 2 );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "C changed with the squared loss" );

  svm.set_squared_loss( false );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "back to the hinge loss" );

  // the targets: the signs, hence the Hessian, hence a restart
  svm.chg_target( - svm.get_y()[ 0 ] , 0 );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "one target flipped" );

  {
   doubleVec flip( 4 );
   for( Index i = 0 ; i < 4 ; ++i )
    flip[ i ] = - svm.get_y()[ 10 + i ];
   svm.chg_targets( flip.begin() , Block::Range( 10 , 14 ) );
   }
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "a range of targets flipped" );

  {
   Block::Subset nms = { 20 , 30 , 40 };
   doubleVec flip( 3 );
   for( Index i = 0 ; i < 3 ; ++i )
    flip[ i ] = - svm.get_y()[ nms[ i ] ];
   svm.chg_targets( flip.begin() , std::move( nms ) , true );
   }
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "a subset of targets flipped" );

  // a target that the concrete class does not admit leaves everything as is
  bool caught = false;
  const double before = svm.get_y()[ 5 ];
  try { svm.chg_target( 0.5 , 5 ); }
  catch( const std::exception & e ) { caught = true; }
  check( caught && ( svm.get_y()[ 5 ] == before ) &&
         ( svm.get_dual_signs()[ 5 ] == before ) ,
         "a rejected target changes nothing" );

  /* Samples added and removed: the dual index space changes size, and the
   * multipliers of the samples that are still there are what the
   * re-optimization starts from. */

  { doubleVec nX , ny;
    make_svc_data( 6 , m , nX , ny , 77 );
    svm.add_samples( 6 , nX , ny );
    }
  check( svm.get_NSamples() == n + 6 , "the samples are there" );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "samples added" );

  svm.remove_samples( Block::Range( 3 , 9 ) );
  check( svm.get_NSamples() == n , "the samples are gone" );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "a range of samples removed" );

  svm.remove_samples( Block::Subset( { 0 , 7 , 13 , 21 } ) , true );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "a subset of samples removed" );

  { doubleVec nX , ny;
    make_svc_data( 4 , m , nX , ny , 91 );
    svm.add_samples( 4 , nX , ny );
    svm.remove_samples( Block::Subset( { 1 , 2 } ) , true );
    }
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "samples added and removed in one go" );

  // the whole data set cannot go, and a rejected removal changes nothing
  { bool caught = false;
    const Index before = svm.get_NSamples();
    Block::Subset all( before );
    std::iota( all.begin() , all.end() , Index( 0 ) );
    try { svm.remove_samples( std::move( all ) , true ); }
    catch( const std::exception & e ) { caught = true; }
    check( caught && ( svm.get_NSamples() == before ) ,
           "removing every sample is refused" );
    }

  // the kernel and the regularisation of the bias: everything changes
  svm.set_kernel( SVMBlock::kGaussian , 0.25 );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "the kernel changed" );

  // ... and the samples still follow, with the Gram matrix of a kernel that
  // is no longer the linear one
  { doubleVec nX , ny;
    make_svc_data( 5 , m , nX , ny , 43 );
    svm.add_samples( 5 , nX , ny );
    }
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "samples added with a nonlinear kernel" );

  svm.set_reg_bias( true );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "the bias regularised" );

  svm.remove_samples( Block::Range( 0 , 5 ) );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "samples removed with the bias regularised" );

  svm.unregister_Solver( solver );
  delete solver;
  }

 // the same, on a regression problem, where epsilon is one more knob - - - -
 {
  const Index n = 40 , m = 3;
  doubleVec X , y;
  make_svr_data( n , m , X , y , 17 );

  SVRBlock svm;
  svm.load( n , m , X , y );

  auto solver = Solver::new_Solver( "SMOSolver" );
  solver->set_par( SMOSolver::dblSMOTol , 1e-10 );
  svm.register_Solver( solver );

  resolve( solver );

  svm.set_epsilon( 0.3 );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "epsilon increased" );

  svm.set_epsilon( 0.01 );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "epsilon decreased" );

  // a target of a regression problem only changes the linear coefficients
  svm.chg_target( svm.get_y()[ 3 ] + 1 , 3 );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "one target changed" );

  svm.set_C( 8 );
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "C increased" );

  svm.unregister_Solver( solver );
  delete solver;
  }

 /* The exact solution path: a sample is learnt by growing its multiplier
  * from zero, and unlearnt by driving it back to zero, keeping every other
  * one at its own optimality condition all along, so that what is left at
  * the end is the exact solution of the problem with, or without, it. */

 std::cout << "the exact solution path" << std::endl;
 {
  const Index n = 40 , m = 3;
  doubleVec X , y;
  make_svc_data( n , m , X , y , 23 );

  SVCBlock svm;
  svm.set_C( 1 );
  svm.load( n , m , X , y );

  auto solver = Solver::new_Solver( "SMOSolver" );
  auto smo = dynamic_cast< SMOSolver * >( solver );
  solver->set_par( SMOSolver::dblSMOTol , 1e-10 );
  svm.register_Solver( solver );

  double value = resolve( solver );

  Index sv , nsv;
  find_support( & svm , sv , nsv );
  check( ( sv < n ) && ( nsv < n ) ,
         "the solution has both a support vector and a sample it ignores" );

  // a sample the model does not lean on is unlearnt for free
  check( smo->unlearn( nsv ) == Solver::kOK ,
         "a sample carrying no multiplier is unlearnt" );
  check( ! smo->get_iter() , "which takes no event at all" );
  check_close( smo->get_var_value() , value , 1e-10 ,
               "and leaves the solution where it was" );

  // a support vector is unlearnt by walking the path, and what is left is
  // the solution of the problem that sample is not part of
  check( smo->unlearn( sv ) == Solver::kOK , "a support vector is unlearnt" );
  check( smo->get_iter() > 0 , "which does take events" );
  check_close( smo->get_var_value() , without_sample( & svm , sv ) , 1e-8 ,
               "and gives exactly the problem without it" );

  // the sample is still in the SVMBlock, and a compute() brings it back
  check_close( resolve( solver ) , value , 1e-8 ,
               "the sample unlearnt is still there, and comes back" );

  // the samples that are added are learnt along the path, and the path gets
  // the answer the iteration gets
  { doubleVec nX , ny;
    make_svc_data( 5 , m , nX , ny , 61 );
    svm.add_samples( 5 , nX , ny );
    }
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "the samples added are learnt along the path" );

  solver->set_par( SMOSolver::intSMOPath , 0 );
  { doubleVec nX , ny;
    make_svc_data( 5 , m , nX , ny , 62 );
    svm.add_samples( 5 , nX , ny );
    }
  check_close( resolve( solver ) , from_scratch( & svm ) , 1e-8 ,
               "and the same ones are learnt without it" );
  solver->set_par( SMOSolver::intSMOPath , 1 );

  // with the squared loss the multipliers have no upper bound, and with the
  // bias regularised there is no equality constraint to keep: the walk is
  // the same one in a different geometry
  svm.set_squared_loss( true );
  value = resolve( solver );
  find_support( & svm , sv , nsv );
  check( smo->unlearn( sv ) == Solver::kOK ,
         "a support vector is unlearnt with the squared loss" );
  check_close( smo->get_var_value() , without_sample( & svm , sv ) , 1e-8 ,
               "and gives the problem without it" );
  svm.set_squared_loss( false );

  svm.set_reg_bias( true );
  value = resolve( solver );
  find_support( & svm , sv , nsv );
  check( smo->unlearn( sv ) == Solver::kOK ,
         "a support vector is unlearnt with the bias regularised" );
  check_close( smo->get_var_value() , without_sample( & svm , sv ) , 1e-8 ,
               "and gives the problem without it" );

  // and it is the same walk with a kernel that is not the linear one
  svm.set_reg_bias( false );
  svm.set_kernel( SVMBlock::kGaussian , 0.5 );
  value = resolve( solver );
  find_support( & svm , sv , nsv );
  check( smo->unlearn( sv ) == Solver::kOK ,
         "a support vector is unlearnt with a nonlinear kernel" );
  check_close( smo->get_var_value() , without_sample( & svm , sv ) , 1e-8 ,
               "and gives the problem without it" );

  svm.unregister_Solver( solver );
  delete solver;
  }

 // the same on a regression problem, where a sample has two multipliers - - -
 {
  const Index n = 30 , m = 3;
  doubleVec X , y;
  make_svr_data( n , m , X , y , 29 );

  SVRBlock svm;
  svm.set_C( 2 );
  svm.load( n , m , X , y );

  auto solver = Solver::new_Solver( "SMOSolver" );
  auto smo = dynamic_cast< SMOSolver * >( solver );
  solver->set_par( SMOSolver::dblSMOTol , 1e-10 );
  svm.register_Solver( solver );

  const double value = resolve( solver );

  Index sv , nsv;
  find_support( & svm , sv , nsv );

  check( smo->unlearn( sv ) == Solver::kOK ,
         "a support vector of a regression is unlearnt" );
  check_close( smo->get_var_value() , without_sample( & svm , sv ) , 1e-8 ,
               "and gives the problem without it" );

  if( nsv < n ) {
   check( smo->unlearn( nsv ) == Solver::kOK ,
          "a sample inside the tube is unlearnt" );
   check( ! smo->get_iter() , "which takes no event" );
   }

  check_close( resolve( solver ) , value , 1e-8 ,
               "and both come back when the problem is solved again" );

  svm.unregister_Solver( solver );
  delete solver;
  }

 // the abstract representation follows the changes - - - - - - - - - - - - -

 std::cout << "changing with the abstract representation" << std::endl;
 {
  const Index n = 30 , m = 3;
  doubleVec Xc , yc , Xr , yr;
  make_svc_data( n , m , Xc , yc , 19 );
  make_svr_data( n , m , Xr , yr , 23 );

  for( int form = SVMBlock::kWolfeDual ; form <= SVMBlock::kPrimal ;
       ++form ) {
   const std::string frm = ( form == SVMBlock::kWolfeDual ) ? "dual: "
                                                            : "primal: ";

   check_change( "SVCBlock" , form , Xc , yc , n , m ,
                 []( SVMBlock * svm ) { svm->set_C( 12 ); } ,
                 frm + "C" );

   check_change( "SVCBlock" , form , Xc , yc , n , m ,
                 []( SVMBlock * svm ) { svm->set_squared_loss( true ); } ,
                 frm + "squared loss" );

   check_change( "SVCBlock" , form , Xc , yc , n , m ,
                 []( SVMBlock * svm ) {
                  svm->set_squared_loss( true );
                  svm->set_C( 0.5 );
                  } , frm + "squared loss and C" );

   check_change( "SVCBlock" , form , Xc , yc , n , m ,
                 []( SVMBlock * svm ) { svm->set_reg_bias( true ); } ,
                 frm + "regularised bias" );

   check_change( "SVCBlock" , form , Xc , yc , n , m ,
                 []( SVMBlock * svm ) {
                  svm->chg_target( - svm->get_y()[ 1 ] , 1 );
                  } , frm + "a target" );

   check_change( "SVRBlock" , form , Xr , yr , n , m ,
                 []( SVMBlock * svm ) {
                  dynamic_cast< SVRBlock * >( svm )->set_epsilon( 0.4 );
                  } , frm + "epsilon" );

   check_change( "SVRBlock" , form , Xr , yr , n , m ,
                 []( SVMBlock * svm ) {
                  svm->chg_target( svm->get_y()[ 2 ] + 1 , 2 );
                  } , frm + "a target of a regression" );

   /* Samples added and removed: the dual index space changes size, hence the
    * abstract representation gains and loses Variable and Constraint rather
    * than being rebuilt, and it still has to be the one it would have been
    * had it been generated once the change was done. */

   check_change( "SVCBlock" , form , Xc , yc , n , m ,
                 [ m ]( SVMBlock * svm ) {
                  doubleVec nX , ny;
                  make_svc_data( 4 , m , nX , ny , 61 );
                  svm->add_samples( 4 , nX , ny );
                  } , frm + "samples added" );

   check_change( "SVCBlock" , form , Xc , yc , n , m ,
                 []( SVMBlock * svm ) {
                  svm->remove_samples( Block::Range( 5 , 9 ) );
                  } , frm + "a range of samples removed" );

   check_change( "SVCBlock" , form , Xc , yc , n , m ,
                 []( SVMBlock * svm ) {
                  svm->remove_samples( Block::Subset( { 0 , 6 , 17 } ) ,
                                       true );
                  } , frm + "a subset of samples removed" );

   check_change( "SVCBlock" , form , Xc , yc , n , m ,
                 [ m ]( SVMBlock * svm ) {
                  doubleVec nX , ny;
                  make_svc_data( 6 , m , nX , ny , 67 );
                  svm->add_samples( 6 , nX , ny );
                  svm->remove_samples( Block::Subset( { 1 , 2 , 30 } ) ,
                                       true );
                  } , frm + "samples added and removed" );

   /* A regression sample is two dual indices, adjacent so that an addition
    * lands at the end of the dual index space and a removal takes away a
    * pair: these two are what tells that the layout is followed. */

   check_change( "SVRBlock" , form , Xr , yr , n , m ,
                 [ m ]( SVMBlock * svm ) {
                  doubleVec nX , ny;
                  make_svr_data( 3 , m , nX , ny , 71 );
                  svm->add_samples( 3 , nX , ny );
                  } , frm + "samples added to a regression" );

   check_change( "SVRBlock" , form , Xr , yr , n , m ,
                 []( SVMBlock * svm ) {
                  svm->remove_samples( Block::Subset( { 4 , 11 } ) , true );
                  } , frm + "samples removed from a regression" );
   }

  // the kernel has no primal, hence it is only changed under the dual
  check_change( "SVCBlock" , SVMBlock::kWolfeDual , Xc , yc , n , m ,
                []( SVMBlock * svm ) {
                 svm->set_kernel( SVMBlock::kGaussian , 0.5 );
                 } , "dual: the kernel" );

  // and it cannot be changed to a nonlinear one under the primal
  bool caught = false;
  try {
   SVCBlock svm;
   svm.load( n , m , Xc , yc );
   SimpleConfiguration< int > cfg( SVMBlock::kPrimal );
   svm.generate_abstract_variables( & cfg );
   svm.set_kernel( SVMBlock::kGaussian );
   }
  catch( const std::exception & e ) { caught = true; }
  check( caught , "a nonlinear kernel is refused under the primal" );
  }

 // the Solution saving the trained model - - - - - - - - - - - - - - - - - -

 std::cout << "SVMBlockSolution" << std::endl;
 {
  doubleVec X , y;
  make_svc_data( 40 , 3 , X , y , 7 );

  SVCBlock svm;
  svm.set_C( 4 );
  svm.load( 40 , 3 , X , y );

  const double value = train( & svm );

  // what get_Solution() gives is dictated by the Configuration: the model
  // only if it is asked for, the abstract representation by default
  SimpleConfiguration< int > model_cfg( 3 );
  auto sol = dynamic_cast< SVMBlockSolution * >(
                                   svm.get_Solution( & model_cfg , false ) );
  check( sol , "the Configuration asks for a SVMBlockSolution" );

  auto other = svm.get_Solution( nullptr , true );
  check( ! dynamic_cast< SVMBlockSolution * >( other ) ,
         "the abstract representation is still the default" );
  delete other;

  if( sol ) {
   check( ( sol->get_alphas() == svm.get_alphas() ) &&
          ( sol->get_b() == svm.get_b() ) , "the model is saved" );

   // the model is restored into a SVMBlock holding the same data set, which
   // therefore predicts exactly the same way
   SVCBlock copy;
   copy.set_C( 4 );
   copy.load( 40 , 3 , X , y );
   sol->write( & copy );

   double worst = 0;
   for( Index i = 0 ; i < 40 ; ++i )
    worst = std::max( worst , std::abs( copy.decision_function( svm.get_x( i )
                                                                ) -
                                        svm.decision_function( svm.get_x( i )
                                                                ) ) );
   check( worst < 1e-12 , "the restored model is the same model" );

   // scale() and sum() are the algebra a Solution has to provide
   auto half = sol->scale( 0.5 );
   half->sum( sol , 0.5 );
   check( ( half->get_alphas() == sol->get_alphas() ) &&
          ( half->get_b() == sol->get_b() ) ,
          "half a model plus half the same model is the model" );
   delete half;

   sol->serialize( "svmsolution.nc4" , true );
   delete sol;
   }

  /* A Solver working on the abstract representation leaves the model in the
   * Variable, and the physical representation knows nothing about it: the
   * Solution has to go and find it there. */
  {
   SVCBlock other;
   other.set_C( 4 );
   other.load( 40 , 3 , X , y );

   SimpleConfiguration< int > primal( SVMBlock::kPrimal );
   other.generate_abstract_variables( & primal );
   other.generate_abstract_constraints();
   other.generate_objective();

   fill_primal( & other , svm.get_w() , svm.get_b() );

   auto asol = dynamic_cast< SVMBlockSolution * >(
                                  other.get_Solution( & model_cfg , false ) );
   check( asol && ( asol->get_w() == svm.get_w() ) &&
          ( asol->get_b() == svm.get_b() ) ,
          "the model is read out of the abstract representation" );
   delete asol;
   }

  auto in = dynamic_cast< SVMBlockSolution * >(
                                  Solution::deserialize( "svmsolution.nc4" ) );
  check( in , "the netCDF file is read back as a SVMBlockSolution" );

  if( in ) {
   SVCBlock copy;
   copy.set_C( 4 );
   copy.load( 40 , 3 , X , y );
   in->write( & copy );

   check_close( primal_value( & copy ) , value , 1e-8 ,
                "same model after the round trip" );
   delete in;
   }
  }

 // the Solution the Solver provides on its own - - - - - - - - - - - - - - -

 std::cout << "Solver::get_Solution" << std::endl;
 {
  const Index n = 40 , m = 3;
  doubleVec X , y;
  make_svc_data( n , m , X , y , 7 );

  // no abstract representation is ever generated here: the whole point of
  // the Solution the Solver provides is that none is needed
  SVCBlock svm;
  svm.set_C( 4 );
  svm.load( n , m , X , y );

  auto solver = Solver::new_Solver( "SMOSolver" );
  solver->set_par( SMOSolver::dblSMOTol , 1e-10 );
  svm.register_Solver( solver );

  check( ! solver->get_Solution() , "no Solution before compute()" );

  solver->compute();

  SimpleConfiguration< int > model_cfg( 3 );
  auto sol = dynamic_cast< SVMBlockSolution * >(
                                       solver->get_Solution( & model_cfg ) );
  check( sol , "the Solver provides the model as a SVMBlockSolution" );

  // and it does so out of its own data: the SVMBlock is left alone, its
  // model being still the all-zero one that load() has left there
  check( svm.get_alphas() == doubleVec( svm.get_NDual() , 0 ) ,
         "the SVMBlock is not written into" );

  // the model the Solver would write into the SVMBlock is the same one
  solver->get_var_solution();
  if( sol )
   check( ( sol->get_alphas() == svm.get_alphas() ) &&
          ( sol->get_b() == svm.get_b() ) && sol->get_w().empty() ,
          "it is the model the Solver has found" );

  /* Any other Solution saves the abstract representation, which only the
   * SVMBlock can fill: it is generated now, and the default Configuration
   * has to give the usual ColVariableSolution. */
  SimpleConfiguration< int > dual( SVMBlock::kWolfeDual );
  svm.generate_abstract_variables( & dual );
  svm.generate_abstract_constraints();
  svm.generate_objective();

  auto other = solver->get_Solution();
  check( dynamic_cast< ColVariableSolution * >( other ) ,
         "any other Solution comes from the SVMBlock" );

  // whatever it is, it has to be the same model
  if( other ) {
   svm.set_dual_solution( doubleVec( svm.get_NDual() , 0 ) , 0 );
   other->write( & svm );
   svm.get_solution_from_abstract();
   check_close( primal_value( & svm ) , solver->get_var_value() , 1e-8 ,
                "the same model, whoever provides the Solution" );
   }

  delete other;
  delete sol;

  svm.unregister_Solver( solver );
  delete solver;
  }

 // LIBSVM, when the module has been built with it - - - - - - - - - - - - -

 /* Solver::new_Solver() throws if the name is not in the factory, which is
  * what happens when the module has been built without LIBSVM: this is the
  * only way of asking whether it is there, the factory itself not being
  * accessible from outside. */

 Solver * probe = nullptr;
 try { probe = Solver::new_Solver( "LIBSVMSolver" ); }
 catch( const std::exception & ) {}

 if( probe ) {
  delete probe;

  std::cout << "LIBSVMSolver" << std::endl;

  /* LIBSVM is an independent implementation of the very algorithm SMOSolver
   * implements, hence the check is that the two agree: same value, same
   * model, and the multipliers LIBSVM is made to give back must be worth,
   * according to the SVMBlock itself, the value the Solver reports. */

  auto same_as_SMO = [ & ]( SVMBlock * svm , const std::string & tag ) {
   const double value = train( svm );          // SMOSolver, tolerance 1e-10
   const doubleVec alpha = svm->get_alphas();
   const double b = svm->get_b();

   /* LIBSVMSolver is only there when the module has been built with LIBSVM,
    * hence everything here goes through the factory and the parameters are
    * set by name, exactly as a configuration file would do. */
   auto solver = Solver::new_Solver( "LIBSVMSolver" );
   solver->set_par( solver->dbl_par_str2idx( "dblLSVMTol" ) , 1e-9 );
   svm->register_Solver( solver );

   const int status = solver->compute();
   check( status == Solver::kOK , "LIBSVM solves, " + tag );

   check_close( solver->get_var_value() , value , 1e-6 ,
                "LIBSVM agrees with SMO on the value, " + tag );

   solver->get_var_solution();

   check_close( svm->dual_objective( svm->get_alphas() ) ,
                solver->get_var_value() , 1e-6 ,
                "the multipliers of LIBSVM are worth what it says, " + tag );

   double dm = std::abs( svm->get_b() - b );
   for( Index k = 0 ; k < alpha.size() ; ++k )
    dm = std::max( dm , std::abs( svm->get_alphas()[ k ] - alpha[ k ] ) );
   check( dm < 1e-5 , "LIBSVM finds the same model as SMO, " + tag );

   // the model also comes out without going through the SVMBlock
   SimpleConfiguration< int > model_cfg( 3 );
   auto sol = dynamic_cast< SVMBlockSolution * >(
                                       solver->get_Solution( & model_cfg ) );
   check( sol && ( sol->get_alphas() == svm->get_alphas() ) &&
          ( sol->get_b() == svm->get_b() ) ,
          "LIBSVM fills the SVMBlockSolution itself, " + tag );
   delete sol;

   svm->unregister_Solver( solver );
   delete solver;
   };

  {
   doubleVec X , y;
   make_svc_data( 60 , 3 , X , y , 1 );

   SVCBlock svm;
   svm.set_kernel( SVMBlock::kLinear );
   svm.set_C( 10 );
   svm.load( 60 , 3 , X , y );
   same_as_SMO( & svm , "linear SVC" );
   check( accuracy( & svm ) == 1 , "LIBSVM separates the data" );

   /* LIBSVM solves the two-class problem with the *first* label it meets in
    * the data set as the positive one, so that its decision function is the
    * opposite of ours whenever that label is -1: the very same data set with
    * the two classes swapped takes the other branch. */
   for( auto & yi : y )
    yi = - yi;

   SVCBlock swapped;
   swapped.set_kernel( SVMBlock::kLinear );
   swapped.set_C( 10 );
   swapped.load( 60 , 3 , X , y );
   same_as_SMO( & swapped , "linear SVC, classes swapped" );
   }

  {
   doubleVec X , y;
   make_xor_data( X , y , 5 );
   const Index n = y.size();

   SVCBlock svm;
   svm.set_kernel( SVMBlock::kGaussian , 1 );
   svm.set_C( 10 );
   svm.load( n , 2 , X , y );
   same_as_SMO( & svm , "Gaussian SVC" );
   }

  {
   doubleVec X , y;
   make_svr_data( 40 , 3 , X , y , 3 );

   SVRBlock svm;
   svm.set_kernel( SVMBlock::kLinear );
   svm.set_C( 8 );
   svm.set_epsilon( 0.1 );
   svm.load( 40 , 3 , X , y );
   same_as_SMO( & svm , "linear SVR" );
   }

  {
   doubleVec X , y;
   make_svr_data( 30 , 2 , X , y , 4 );

   SVRBlock svm;
   svm.set_kernel( SVMBlock::kPoly , 0.5 , 3 , 1 );
   svm.set_C( 4 );
   svm.set_epsilon( 0.2 );
   svm.load( 30 , 2 , X , y );
   same_as_SMO( & svm , "polynomial SVR" );
   }

  /* What LIBSVM cannot be asked has to be refused, and loudly: an
   * approximation of the training problem that has been asked for would be
   * worse than an error. */

  auto refuses = [ & ]( SVMBlock * svm , const std::string & tag ) {
   auto solver = Solver::new_Solver( "LIBSVMSolver" );
   bool thrown = false;
   try {
    solver->set_Block( svm );
    solver->compute();
    }
   catch( const std::exception & ) { thrown = true; }
   check( thrown , "LIBSVM refuses " + tag );
   solver->set_Block( nullptr );
   delete solver;
   };

  {
   doubleVec X , y;
   make_svc_data( 20 , 2 , X , y , 9 );

   SVCBlock sl;
   sl.set_C( 1 );
   sl.set_squared_loss( true );
   sl.load( 20 , 2 , X , y );
   refuses( & sl , "the squared loss" );

   SVCBlock rb;
   rb.set_C( 1 );
   rb.set_reg_bias( true );
   rb.load( 20 , 2 , X , y );
   refuses( & rb , "the regularised bias" );

   SVCBlock rw;
   rw.set_C( 1 );
   rw.set_reg_weight( 0.5 );
   rw.load( 20 , 2 , X , y );
   refuses( & rw , "a weight on the regularisation term" );

   SVCBlock lp;
   lp.set_C( 1 );
   lp.set_kernel( SVMBlock::kLaplacian , 1 );
   lp.load( 20 , 2 , X , y );
   refuses( & lp , "the Laplacian kernel" );
   }
  }

 // the netCDF round trip - - - - - - - - - - - - - - - - - - - - - - - - - -

 std::cout << "serialization" << std::endl;
 {
  doubleVec X , y;
  make_svr_data( 20 , 4 , X , y , 11 );

  SVRBlock out;
  out.set_kernel( SVMBlock::kPoly , 0.5 , 4 , 2 );
  out.set_C( 7 );
  out.set_epsilon( 0.25 );
  out.set_squared_loss( true );
  out.set_reg_bias( true );
  out.load( 20 , 4 , X , y );

  out.serialize( "svmblock.nc4" , eBlockFile );

  auto in = dynamic_cast< SVRBlock * >( Block::deserialize( "svmblock.nc4" ) );
  check( in , "the netCDF file is read back as a SVRBlock" );

  if( in ) {
   check( ( in->get_NSamples() == 20 ) && ( in->get_NFeatures() == 4 ) &&
          ( in->get_X() == X ) && ( in->get_y() == y ) , "same data set" );
   check( ( in->get_C() == 7 ) && ( in->get_epsilon() == 0.25 ) &&
          ( in->get_kernel_type() == SVMBlock::kPoly ) &&
          ( in->get_gamma() == 0.5 ) && ( in->get_degree() == 4 ) &&
          ( in->get_coef0() == 2 ) && in->get_squared_loss() &&
          in->get_reg_bias() , "same hyper-parameters" );

   check_close( train( in ) , train( & out ) , 1e-8 ,
                "same optimal value after the round trip" );
   delete in;
   }
  }

 // the result- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

 if( failed )
  std::cout << std::endl << "SVMBlock: " << failed << " test(s) FAILED"
            << std::endl;
 else
  std::cout << std::endl << "SVMBlock: all tests passed" << std::endl;

 return( failed ? 1 : 0 );

 }  // end( main )

/*--------------------------------------------------------------------------*/
/*---------------------------- End File test.cpp ---------------------------*/
/*--------------------------------------------------------------------------*/
