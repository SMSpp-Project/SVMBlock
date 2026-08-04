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
 * maximisation that strong duality makes equal to the primal. The decomposed formulation is checked both structurally, against
 * what a generic Lagrangian Solver requires, and numerically, by verifying
 * that at the optimum of the monolithic problem the consensus constraints are
 * satisfied and the sub-Block objectives add up to the monolithic value.
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

#include <cmath>

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

 double v = 0;
 for( Index i = 0 ; i < n ; ++i )
  for( Index j = 0 ; j < n ; ++j )
   v += c[ i ] * c[ j ] * K[ std::size_t( i ) * n + j ];
 v /= 2;

 if( svm->get_reg_bias() )
  v += b * b / 2;

 doubleVec f( n , b );
 for( Index i = 0 ; i < n ; ++i )
  for( Index j = 0 ; j < n ; ++j )
   f[ i ] += c[ j ] * K[ std::size_t( j ) * n + i ];

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
 auto xv = svm->get_static_variable_v< ColVariable >( "xi" );
 auto bv = svm->get_static_variable< ColVariable >( "b" );

 const Index m = svm->get_NFeatures();

 for( Index j = 0 ; j < m ; ++j )
  (*wv)[ j ].set_value( w[ j ] );

 bv->set_value( b );

 auto & s = svm->get_dual_signs();
 auto & di = svm->get_dual_samples();
 auto & q = svm->get_dual_costs();
 auto & X = svm->get_X();

 for( Index k = 0 ; k < svm->get_NDual() ; ++k ) {
  double f = b;
  for( Index j = 0 ; j < m ; ++j )
   f += w[ j ] * X[ std::size_t( di[ k ] ) * m + j ];
  (*xv)[ k ].set_value( std::max( double( 0 ) , - q[ k ] - s[ k ] * f ) );
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
  auto av = svm->get_static_variable_v< ColVariable >( "alpha" );
  auto & alpha = svm->get_alphas();
  for( Index k = 0 ; k < N ; ++k )
   (*av)[ k ].set_value( alpha[ k ] );
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

 // the decomposed formulation - - - - - - - - - - - - - - - - - - - - - - - -

 std::cout << "decomposed formulation" << std::endl;
 {
  const Index n = 48 , m = 3 , P = 4;

  doubleVec X , y;
  make_svc_data( n , m , X , y , 5 );

  // the reference: the same problem solved as a whole
  SVCBlock ref;
  ref.set_kernel( SVMBlock::kLinear );
  ref.set_C( 3 );
  ref.load( n , m , X , y );

  const double primal = train( & ref );
  const auto w = ref.get_w();
  const double b = ref.get_b();

  SVCBlock dec;
  dec.set_kernel( SVMBlock::kLinear );
  dec.set_C( 3 );
  dec.load( n , m , X , y );

  // the formulation, and the number of chunks it is decomposed into, are a
  // Configuration matter, not part of the data of the SVMBlock
  SimpleConfiguration< std::pair< int , int > >
   cfg( { SVMBlock::kDecomposed , int( P ) } );

  dec.generate_abstract_variables( & cfg );
  dec.generate_abstract_constraints();
  dec.generate_objective();

  // the structure a generic Lagrangian Solver expects
  check( ! dec.get_static_variable< ColVariable >( "w" ) &&
         ! dec.get_static_variable_v< ColVariable >( "alpha" ) ,
         "no Variable in the father Block" );
  check( dec.get_number_nested_Blocks() == P , "one sub-Block per chunk" );
  check( dec.get_objective() &&
         ( dec.get_objective()->get_num_active_var() == 0 ) ,
         "the Objective of the father Block is empty" );

  auto link = dec.get_static_constraint_v< FRowConstraint >( "link" );
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
   auto sub = dynamic_cast< SVMBlock * >( dec.get_nested_Block( p ) );
   tot += sub->get_NSamples();
   auto & ys = sub->get_y();
   if( std::count( ys.begin() , ys.end() , 1. ) == 0 ||
       std::count( ys.begin() , ys.end() , -1. ) == 0 )
    both = false;
   }
  check( tot == n , "the chunks partition the samples" );
  check( both , "every chunk sees both classes" );

  // the reformulation is exact: at the optimum of the monolithic problem the
  // consensus constraints are satisfied and the sub-Block objectives add up
  // to the monolithic primal value
  double sum = 0;
  for( Index p = 0 ; p < P ; ++p ) {
   auto sub = dynamic_cast< SVMBlock * >( dec.get_nested_Block( p ) );
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
  check_close( sum , primal , 1e-8 , "the reformulation is exact" );

  dec.get_solution_from_abstract();
  double dw = std::abs( dec.get_b() - b );
  auto w2 = dec.get_w();
  for( Index j = 0 ; j < m ; ++j )
   dw = std::max( dw , std::abs( w2[ j ] - w[ j ] ) );
  check( dw < 1e-12 , "the model is recovered out of the sub-Block" );

  // a chunk of a single class would have an unbounded subproblem
  bool caught = false;
  try {
   SVCBlock bad;
   bad.set_kernel( SVMBlock::kLinear );
   bad.load( n , m , X , y );
   SimpleConfiguration< std::pair< int , int > >
    bcfg( { SVMBlock::kDecomposed , int( n ) } );
   bad.generate_abstract_variables( & bcfg );
   }
  catch( const std::exception & e ) { caught = true; }
  check( caught , "single-class chunks are refused" );
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
