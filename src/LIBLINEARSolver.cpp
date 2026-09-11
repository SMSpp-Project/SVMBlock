/*--------------------------------------------------------------------------*/
/*----------------------- File LIBLINEARSolver.cpp -------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Implementation of the LIBLINEARSolver class.
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

#include "LIBLINEARSolver.h"

#include "SVCBlock.h"

#include "SVRBlock.h"

#include <algorithm>

#include <cmath>

#include <cstdio>

#include <mutex>

#include <string>

/*--------------------------------------------------------------------------*/
/*------------------------- NAMESPACE AND USING ----------------------------*/
/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

/*--------------------------------------------------------------------------*/
/*----------------------------- STATIC MEMBERS -----------------------------*/
/*--------------------------------------------------------------------------*/

// register LIBLINEARSolver in the Solver factory

SMSpp_insert_in_factory_cpp_1( LIBLINEARSolver );

/*--------------------------------------------------------------------------*/
/*------------------------------- FUNCTIONS --------------------------------*/
/*--------------------------------------------------------------------------*/

/* Whether LIBLINEAR has stopped on its cap on the number of iterations is
 * only said in the log, so the log is where it is read: the hook below scans
 * each line for the warning and prints it only if asked to. Both the hook and
 * the flag it writes are of the translation unit rather than of the object,
 * LIBLINEAR having one print function for the whole library and no way of
 * carrying anything of the caller into it, and the mutex is what makes the
 * pair safe when two LIBLINEARSolver train at the same time. */

static std::mutex train_mutex;
static bool s_capped = false;
static bool s_verbose = false;

static void sniff( const char * line )
{
 if( std::string( line ).find( "reaching max number of iterations" ) !=
     std::string::npos )
  s_capped = true;

 if( s_verbose ) {
  std::fputs( line , stdout );
  std::fflush( stdout );
  }
 }

/*--------------------------------------------------------------------------*/
/*--------------------- DERIVED METHODS OF BASE CLASS ----------------------*/
/*--------------------------------------------------------------------------*/

void LIBLINEARSolver::set_Block( Block * block )
{
 if( block == f_Block )  // actually doing nothing
  return;                // cowardly and silently return

 Solver::set_Block( block );

 free_model();
 f_solved = false;
 f_dirty = true;
 f_capped = false;
 f_value = 0;
 f_b = 0;
 v_w.clear();

 if( ! block ) {
  f_SVM = nullptr;
  return;
  }

 f_SVM = dynamic_cast< SVMBlock * >( block );
 if( ! f_SVM )
  throw( std::invalid_argument( "LIBLINEARSolver::set_Block: the Block is "
                                "not a SVMBlock" ) );

 // fail here rather than at the first compute() for whatever is already
 // known not to be a training problem LIBLINEAR can be asked
 check_supported();

 }  // end( LIBLINEARSolver::set_Block )

/*--------------------------------------------------------------------------*/

int LIBLINEARSolver::compute( bool changedvars )
{
 if( ! f_SVM )
  throw( std::logic_error( "LIBLINEARSolver::compute: no SVMBlock is set" ) );

 if( f_solved && ( ! f_dirty ) )   // nothing has changed since the last
  return( f_capped ? kLowPrecision : kOK );          // call: nothing to do

 lock();  // lock the mutex

 /* LIBLINEAR trains from scratch: the data set and the hyper-parameters are
  * re-read out of the SVMBlock each time, so whatever has changed in the
  * meanwhile is dealt with by doing so, and the hyper-parameters have to be
  * checked again since they may have changed since set_Block(). */

 try {
  check_supported();
  build_problem();
  build_parameters();
  }
 catch( ... ) {
  unlock();
  throw;
  }

 if( auto err = check_parameter( & f_prob , & f_par ) ) {
  const std::string msg( err );
  unlock();
  throw( std::invalid_argument( "LIBLINEARSolver::compute: LIBLINEAR "
                                "rejected the problem: " + msg ) );
  }

 f_solved = false;
 f_capped = false;
 free_model();

 {
  std::lock_guard< std::mutex > guard( train_mutex );
  s_capped = false;
  s_verbose = f_log_verb;
  set_print_string_function( & sniff );
  f_model = train( & f_prob , & f_par );
  f_capped = s_capped;
  }

 if( ! f_model ) {
  unlock();
  return( kError );
  }

 try {
  extract_model();
  }
 catch( ... ) {
  unlock();
  throw;
  }

 f_solved = true;
 f_dirty = false;

 unlock();  // unlock the mutex

 return( f_capped ? kLowPrecision : kOK );

 }  // end( LIBLINEARSolver::compute )

/*--------------------------------------------------------------------------*/

void LIBLINEARSolver::get_var_solution( Configuration * solc )
{
 if( ! f_solved )
  throw( std::logic_error( "LIBLINEARSolver::get_var_solution: no solution "
                           "is available" ) );

 auto w = v_w;  // the Solver keeps its own copy
 f_SVM->set_primal_solution( std::move( w ) , f_b );

 // the model is now in the SVMBlock; leave it also where any Solver working
 // on the abstract representation would have left it, if there is one
 f_SVM->set_solution_in_abstract();

 }  // end( LIBLINEARSolver::get_var_solution )

/*--------------------------------------------------------------------------*/

Solution * LIBLINEARSolver::get_Solution( Configuration * solc )
{
 if( ! f_solved )
  return( nullptr );

 // which Solution the Configuration asks for is the SVMBlock's business,
 // and asking it for an empty one is how it is found out

 auto sol = f_SVM->get_Solution( solc , true );

 if( auto msol = dynamic_cast< SVMBlockSolution * >( sol ) ) {
  msol->set_primal_model( doubleVec( v_w ) , f_b );
  return( msol );
  }

 delete sol;

 // anything else saves the abstract representation, which this Solver does
 // not write into: the base class does the only thing that can be done

 return( Solver::get_Solution( solc ) );

 }  // end( LIBLINEARSolver::get_Solution )

/*--------------------------------------------------------------------------*/
/*--------------------------- PROTECTED METHODS ----------------------------*/
/*--------------------------------------------------------------------------*/

void LIBLINEARSolver::check_supported( void ) const
{
 static const std::string _prfx = "LIBLINEARSolver::check_supported: ";

 if( f_SVM->get_kernel_type() != SVMBlock::kLinear )
  throw( std::invalid_argument( _prfx + "LIBLINEAR only has the linear "
                                "kernel" ) );

 if( ! f_SVM->get_reg_bias() )
  throw( std::invalid_argument( _prfx + "LIBLINEAR only has the regularised "
                                "bias, the bias being one more component of "
                                "the model" ) );

 if( f_SVM->has_linear_term() )
  throw( std::invalid_argument( _prfx + "LIBLINEAR has no linear term in the "
                                "primal, hence it cannot solve the "
                                "subproblem of a chunk" ) );

 }  // end( LIBLINEARSolver::check_supported )

/*--------------------------------------------------------------------------*/

void LIBLINEARSolver::build_problem( void )
{
 const Index n = f_SVM->get_NSamples();
 const Index m = f_SVM->get_NFeatures();

 /* The samples are handed over as the sparse rows LIBLINEAR wants: the zero
  * features are dropped and each row is terminated by the index -1, the
  * indices being 1-based. The bias is not a datum of LIBLINEAR but one more
  * feature, of index m + 1 and value one, which the caller appends: the
  * weight that goes with it is the bias of the model, and it is regularised
  * as every other weight is. */

 v_node.clear();
 v_node.reserve( std::size_t( n ) * ( m + 2 ) );

 std::vector< std::size_t > start( n );

 for( Index i = 0 ; i < n ; ++i ) {
  start[ i ] = v_node.size();
  auto xi = f_SVM->get_x( i );
  for( Index j = 0 ; j < m ; ++j )
   if( xi[ j ] != 0 )
    v_node.push_back( feature_node{ int( j ) + 1 , xi[ j ] } );
  v_node.push_back( feature_node{ int( m ) + 1 , 1 } );
  v_node.push_back( feature_node{ -1 , 0 } );
  }

 // the pointers are taken only now, the vector having grown in between
 v_row.resize( n );
 for( Index i = 0 ; i < n ; ++i )
  v_row[ i ] = v_node.data() + start[ i ];

 auto & y = f_SVM->get_y();
 v_target.assign( y.begin() , y.begin() + n );

 f_prob.l = int( n );
 f_prob.n = int( m ) + 1;   // the bias is the last feature
 f_prob.y = v_target.data();
 f_prob.x = v_row.data();
 f_prob.bias = 1;

 }  // end( LIBLINEARSolver::build_problem )

/*--------------------------------------------------------------------------*/

void LIBLINEARSolver::build_parameters( void )
{
 const bool squared = f_SVM->get_squared_loss();

 if( dynamic_cast< SVCBlock * >( f_SVM ) ) {
  f_par.solver_type = squared ? L2R_L2LOSS_SVC : L2R_L1LOSS_SVC_DUAL;
  f_par.p = 0;
  }
 else
  if( auto svr = dynamic_cast< SVRBlock * >( f_SVM ) ) {
   f_par.solver_type = squared ? L2R_L2LOSS_SVR : L2R_L1LOSS_SVR_DUAL;
   f_par.p = svr->get_epsilon();
   }
  else
   throw( std::invalid_argument( "LIBLINEARSolver::build_parameters: "
                                 "LIBLINEAR has nothing to train a " +
                                 f_SVM->classname() + " with" ) );

 /* Dividing the objective of the SVMBlock by the weight of its regularisation
  * term leaves the same minimiser and gives the objective of LIBLINEAR: the
  * two losses are the same one, C xi and C xi^2, so nothing else is needed. */

 f_par.C = f_SVM->get_C() / f_SVM->get_reg_weight();
 f_par.eps = f_tol;

 f_par.nr_weight = 0;       // no per-class weight on C
 f_par.weight_label = nullptr;
 f_par.weight = nullptr;
 f_par.init_sol = nullptr;

 #if defined( LIBLINEAR_VERSION ) && ( LIBLINEAR_VERSION >= 240 )
  /* Since 2.40 LIBLINEAR can leave the bias out of the regularisation, and a
   * zeroed parameter would ask exactly that: what is wanted here is the
   * opposite, the bias being a component of the model like any other. The
   * one-class parameter arrived with the same version, and check_parameter()
   * looks at it whatever the solver type is. */
  f_par.regularize_bias = 1;
  f_par.nu = 0.5;
 #endif

 }  // end( LIBLINEARSolver::build_parameters )

/*--------------------------------------------------------------------------*/

void LIBLINEARSolver::extract_model( void )
{
 const Index n = f_SVM->get_NSamples();
 const Index m = f_SVM->get_NFeatures();

 /* The bias is not counted among the features of the model, but its weight
  * is the last of them: w has m + 1 entries and nr_feature is m. */

 if( f_model->nr_feature != int( m ) )
  throw( std::logic_error( "LIBLINEARSolver::extract_model: LIBLINEAR has "
                           "not the features of the data set" ) );

 /* LIBLINEAR orders the classes by the label it meets first in the data set,
  * and the decision function it returns is the one of the class its label[ 0 ]
  * names: a data set whose first sample is a negative one therefore comes
  * back mirrored, and the model has to be taken with the opposite sign. */

 const double sign = ( f_model->label && ( f_model->label[ 0 ] != 1 ) ) ? -1
                                                                       : 1;

 v_w.resize( m );
 for( Index j = 0 ; j < m ; ++j )
  v_w[ j ] = sign * f_model->w[ j ];

 f_b = sign * f_model->w[ m ];   // the weight of the appended feature

 /* The value that a SVMBlock reports is the one of its training problem, and
  * this is the one place where it is computed here rather than being read
  * out of the library: LIBLINEAR returns the model and nothing else. */

 double reg = 0;
 for( auto wj : v_w )
  reg += wj * wj;
 reg += f_b * f_b;   // the bias is regularised, check_supported() saw to it

 const bool squared = f_SVM->get_squared_loss();
 const auto svr = dynamic_cast< SVRBlock * >( f_SVM );
 const double eps = svr ? svr->get_epsilon() : 0;
 auto & y = f_SVM->get_y();

 double loss = 0;
 for( Index i = 0 ; i < n ; ++i ) {
  auto xi = f_SVM->get_x( i );
  double dec = f_b;
  for( Index j = 0 ; j < m ; ++j )
   dec += v_w[ j ] * xi[ j ];

  const double slack = svr ? std::max( double( 0 ) ,
                                       std::abs( y[ i ] - dec ) - eps )
                           : std::max( double( 0 ) , 1 - y[ i ] * dec );

  loss += squared ? slack * slack : slack;
  }

 f_value = f_SVM->get_reg_weight() * reg / 2 + f_SVM->get_C() * loss;

 }  // end( LIBLINEARSolver::extract_model )

/*--------------------------------------------------------------------------*/
/*---------------------- End File LIBLINEARSolver.cpp ----------------------*/
/*--------------------------------------------------------------------------*/
