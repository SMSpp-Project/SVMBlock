/*--------------------------------------------------------------------------*/
/*------------------------- File LIBSVMSolver.cpp --------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Implementation of the LIBSVMSolver class.
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

#include "LIBSVMSolver.h"

#include "SVCBlock.h"

#include "SVRBlock.h"

#include <algorithm>

#include <string>

/*--------------------------------------------------------------------------*/
/*------------------------- NAMESPACE AND USING ----------------------------*/
/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

/*--------------------------------------------------------------------------*/
/*----------------------------- STATIC MEMBERS -----------------------------*/
/*--------------------------------------------------------------------------*/

// register LIBSVMSolver in the Solver factory

SMSpp_insert_in_factory_cpp_1( LIBSVMSolver );

/*--------------------------------------------------------------------------*/
/*------------------------------- FUNCTIONS --------------------------------*/
/*--------------------------------------------------------------------------*/

/// what LIBSVM prints its log with when it has to print nothing

static void silence( const char * ) {}

/*--------------------------------------------------------------------------*/
/*--------------------- DERIVED METHODS OF BASE CLASS ----------------------*/
/*--------------------------------------------------------------------------*/

void LIBSVMSolver::set_Block( Block * block )
{
 if( block == f_Block )  // actually doing nothing
  return;                // cowardly and silently return

 Solver::set_Block( block );

 free_model();
 f_solved = false;
 f_dirty = true;
 f_value = 0;
 f_b = 0;
 v_alpha.clear();

 if( ! block ) {
  f_SVM = nullptr;
  return;
  }

 f_SVM = dynamic_cast< SVMBlock * >( block );
 if( ! f_SVM )
  throw( std::invalid_argument( "LIBSVMSolver::set_Block: the Block is not "
                                "a SVMBlock" ) );

 // fail here rather than at the first compute() for whatever is already
 // known not to be a training problem LIBSVM can be asked
 check_supported();

 }  // end( LIBSVMSolver::set_Block )

/*--------------------------------------------------------------------------*/

int LIBSVMSolver::compute( bool changedvars )
{
 if( ! f_SVM )
  throw( std::logic_error( "LIBSVMSolver::compute: no SVMBlock is set" ) );

 if( f_solved && ( ! f_dirty ) )   // nothing has changed since the last
  return( kOK );                  // call, hence there is nothing to do

 lock();  // lock the mutex

 /* LIBSVM trains from scratch: the data set and the hyper-parameters are
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

 if( auto err = svm_check_parameter( & f_prob , & f_par ) ) {
  const std::string msg( err );
  unlock();
  throw( std::invalid_argument( "LIBSVMSolver::compute: LIBSVM rejected the "
                                "problem: " + msg ) );
  }

 svm_set_print_string_function( f_log_verb ? nullptr : & silence );

 f_solved = false;
 free_model();
 f_model = svm_train( & f_prob , & f_par );

 if( ! f_model ) {
  unlock();
  return( kError );
  }

 /* A two-class problem has to come out of LIBSVM as such: it does not if all
  * the samples belong to the same class, in which case there is no model to
  * speak of and LIBSVM silently returns a one-class one. */
 if( ( f_par.svm_type == C_SVC ) && ( f_model->nr_class != 2 ) ) {
  unlock();
  throw( std::invalid_argument( "LIBSVMSolver::compute: the samples all "
                                "belong to the same class" ) );
  }

 extract_model();

 f_solved = true;
 f_dirty = false;

 unlock();  // unlock the mutex

 return( kOK );

 }  // end( LIBSVMSolver::compute )

/*--------------------------------------------------------------------------*/

void LIBSVMSolver::get_var_solution( Configuration * solc )
{
 if( ! f_solved )
  throw( std::logic_error( "LIBSVMSolver::get_var_solution: no solution is "
                           "available" ) );

 auto alpha = v_alpha;  // the Solver keeps its own copy
 f_SVM->set_dual_solution( std::move( alpha ) , f_b );

 // the model is now in the SVMBlock; leave it also where any Solver working
 // on the abstract representation would have left it, if there is one
 f_SVM->set_solution_in_abstract();

 }  // end( LIBSVMSolver::get_var_solution )

/*--------------------------------------------------------------------------*/

Solution * LIBSVMSolver::get_Solution( Configuration * solc )
{
 if( ! f_solved )
  return( nullptr );

 // which Solution the Configuration asks for is the SVMBlock's business,
 // and asking it for an empty one is how it is found out

 auto sol = f_SVM->get_Solution( solc , true );

 if( auto msol = dynamic_cast< SVMBlockSolution * >( sol ) ) {
  msol->set_dual_model( doubleVec( v_alpha ) , f_b );
  return( msol );
  }

 delete sol;

 // anything else saves the abstract representation, which this Solver does
 // not write into: the base class does the only thing that can be done

 return( Solver::get_Solution( solc ) );

 }  // end( LIBSVMSolver::get_Solution )

/*--------------------------------------------------------------------------*/
/*--------------------------- PROTECTED METHODS ----------------------------*/
/*--------------------------------------------------------------------------*/

void LIBSVMSolver::check_supported( void ) const
{
 static const std::string _prfx = "LIBSVMSolver::check_supported: ";

 if( f_SVM->get_squared_loss() )
  throw( std::invalid_argument( _prfx + "LIBSVM only has the linear loss, "
                                "the slacks cannot be squared" ) );

 if( f_SVM->get_reg_bias() )
  throw( std::invalid_argument( _prfx + "LIBSVM always has the equality "
                                "constraint of the unregularised bias" ) );

 if( f_SVM->get_reg_weight() != 1 )
  throw( std::invalid_argument( _prfx + "LIBSVM has no weight on the "
                                "regularisation term" ) );

 if( f_SVM->has_linear_term() )
  throw( std::invalid_argument( _prfx + "LIBSVM has no linear term in the "
                                "primal, hence it cannot solve the "
                                "subproblem of a chunk" ) );

 if( f_SVM->get_kernel_type() == SVMBlock::kLaplacian )
  throw( std::invalid_argument( _prfx + "LIBSVM has no Laplacian kernel" ) );

 }  // end( LIBSVMSolver::check_supported )

/*--------------------------------------------------------------------------*/

void LIBSVMSolver::build_problem( void )
{
 const Index n = f_SVM->get_NSamples();
 const Index m = f_SVM->get_NFeatures();

 /* The samples are handed over as the sparse rows LIBSVM wants: the zero
  * features are dropped and each row is terminated by the index -1, the
  * indices being 1-based. LIBSVM does not copy them, the support vectors of
  * the model pointing into them, hence they are kept here. */

 v_node.clear();
 v_node.reserve( std::size_t( n ) * ( m + 1 ) );

 std::vector< std::size_t > start( n );

 for( Index i = 0 ; i < n ; ++i ) {
  start[ i ] = v_node.size();
  auto xi = f_SVM->get_x( i );
  for( Index j = 0 ; j < m ; ++j )
   if( xi[ j ] != 0 )
    v_node.push_back( svm_node{ int( j ) + 1 , xi[ j ] } );
  v_node.push_back( svm_node{ -1 , 0 } );
  }

 // the pointers are taken only now, the vector having grown in between
 v_row.resize( n );
 for( Index i = 0 ; i < n ; ++i )
  v_row[ i ] = v_node.data() + start[ i ];

 auto & y = f_SVM->get_y();
 v_target.assign( y.begin() , y.begin() + n );

 f_prob.l = int( n );
 f_prob.y = v_target.data();
 f_prob.x = v_row.data();

 }  // end( LIBSVMSolver::build_problem )

/*--------------------------------------------------------------------------*/

void LIBSVMSolver::build_parameters( void )
{
 if( dynamic_cast< SVCBlock * >( f_SVM ) ) {
  f_par.svm_type = C_SVC;
  f_par.p = 0;
  }
 else
  if( auto svr = dynamic_cast< SVRBlock * >( f_SVM ) ) {
   f_par.svm_type = EPSILON_SVR;
   f_par.p = svr->get_epsilon();
   }
  else
   throw( std::invalid_argument( "LIBSVMSolver::build_parameters: LIBSVM has "
                                 "nothing to train a " + f_SVM->classname() +
                                 " with" ) );

 switch( f_SVM->get_kernel_type() ) {
  case( SVMBlock::kLinear ):   f_par.kernel_type = LINEAR; break;
  case( SVMBlock::kPoly ):     f_par.kernel_type = POLY; break;
  case( SVMBlock::kGaussian ): f_par.kernel_type = RBF; break;
  case( SVMBlock::kSigmoid ):  f_par.kernel_type = SIGMOID; break;
  default:
   throw( std::invalid_argument( "LIBSVMSolver::build_parameters: LIBSVM has "
                                 "not this kernel" ) );
  }

 // the gamma of the SVMBlock is the one derived from the data set when it is
 // one of the conventional values, which is what LIBSVM has to be given
 f_par.degree = f_SVM->get_degree();
 f_par.gamma = f_SVM->get_gamma();
 f_par.coef0 = f_SVM->get_coef0();

 f_par.C = f_SVM->get_C();
 f_par.eps = f_tol;
 f_par.cache_size = f_cache;
 f_par.shrinking = f_shrink;
 f_par.probability = 0;

 f_par.nu = 0.5;        // unused, but LIBSVM checks that it is in ( 0 , 1 ]
 f_par.nr_weight = 0;       // no per-class weight on C
 f_par.weight_label = nullptr;
 f_par.weight = nullptr;

 }  // end( LIBSVMSolver::build_parameters )

/*--------------------------------------------------------------------------*/

void LIBSVMSolver::extract_model( void )
{
 const Index n = f_SVM->get_NSamples();
 const Index N = f_SVM->get_NDual();

 /* LIBSVM orders the classes by the label it meets first in the data set,
  * save that for a two-class problem with labels -1 and +1 it swaps them so
  * that +1 is the positive class; since the targets of a SVCBlock are
  * exactly -1 and +1, its decision function is therefore ours and not its
  * opposite. That is checked rather than assumed: were LIBSVM to change its
  * mind, the model would come out mirrored and nothing would say so. */

 if( f_model->label && ( f_model->label[ 0 ] != 1 ) )
  throw( std::logic_error( "LIBSVMSolver::extract_model: LIBSVM has taken "
                           "the negative class as the positive one" ) );

 // the coefficients of the kernel expansion, which is what LIBSVM returns
 doubleVec c( n , 0 );
 for( int t = 0 ; t < f_model->l ; ++t )
  c[ f_model->sv_indices[ t ] - 1 ] = f_model->sv_coef[ 0 ][ t ];

 f_b = - f_model->rho[ 0 ];

 /* The multipliers: c_i is the sum over the dual indices k referring to the
  * sample i of s_k alpha_k, and at most one of those alpha is nonzero at an
  * optimal solution, hence the positive part of c_i is the multiplier of the
  * dual index with positive sign and the negative part that of the other. */

 auto & s = f_SVM->get_dual_signs();
 auto & di = f_SVM->get_dual_samples();

 v_alpha.assign( N , 0 );
 for( Index k = 0 ; k < N ; ++k ) {
  const double ci = c[ di[ k ] ];
  v_alpha[ k ] = ( s[ k ] > 0 ) ? std::max( ci , double( 0 ) )
                                : std::max( - ci , double( 0 ) );
  }

 /* The value of the dual, i.e., - q^T alpha - 1/2 alpha^T Q alpha. With
  * neither the squared loss nor the regularised bias, both of which are
  * ruled out [see check_supported()], alpha^T Q alpha is c^T K c, and
  * c^T K c is sum_i c_i ( f( x_i ) - b ) with f the decision function LIBSVM
  * evaluates itself. Only the support vectors have c_i != 0, so this costs a
  * number of kernel evaluations quadratic in *their* number rather than in
  * the number of samples: computing it out of the Gram matrix would defeat
  * the purpose of using LIBSVM. */

 auto & q = f_SVM->get_dual_costs();
 double qa = 0;
 for( Index k = 0 ; k < N ; ++k )
  qa += q[ k ] * v_alpha[ k ];

 double cKc = 0;
 for( int t = 0 ; t < f_model->l ; ++t ) {
  double dec;
  svm_predict_values( f_model , f_model->SV[ t ] , & dec );
  cKc += c[ f_model->sv_indices[ t ] - 1 ] * ( dec - f_b );
  }

 f_value = - qa - cKc / 2;

 }  // end( LIBSVMSolver::extract_model )

/*--------------------------------------------------------------------------*/
/*----------------------- End File LIBSVMSolver.cpp ------------------------*/
/*--------------------------------------------------------------------------*/
