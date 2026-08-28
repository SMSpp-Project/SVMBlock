/*--------------------------------------------------------------------------*/
/*-------------------------- File LIBSVMSolver.h ---------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Header file for the *concrete* class LIBSVMSolver, which implements the
 * Solver interface for a SVMBlock by handing the training problem over to
 * LIBSVM.
 *
 * \author Donato Meoli \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 *
 * \copyright &copy; by Donato Meoli
 */
/*--------------------------------------------------------------------------*/
/*----------------------------- DEFINITIONS --------------------------------*/
/*--------------------------------------------------------------------------*/

#ifndef __LIBSVMSolver
 #define __LIBSVMSolver
                      /* self-identification: #endif at the end of the file */

/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include "SVMBlock.h"

#include "Solver.h"

#include <libsvm/svm.h>

/*--------------------------------------------------------------------------*/
/*------------------------------ NAMESPACE ---------------------------------*/
/*--------------------------------------------------------------------------*/

/// namespace for the Structured Modeling System++ (SMS++)
namespace SMSpp_di_unipi_it
{
/*--------------------------------------------------------------------------*/
/*-------------------------------- CLASSES ---------------------------------*/
/*--------------------------------------------------------------------------*/
/** @defgroup LIBSVMSolver_CLASSES Classes in LIBSVMSolver.h
 *  @{ */

/*--------------------------------------------------------------------------*/
/*--------------------------- CLASS LIBSVMSolver ---------------------------*/
/*--------------------------------------------------------------------------*/
/*--------------------------- GENERAL NOTES --------------------------------*/
/*--------------------------------------------------------------------------*/
/// a Solver for a SVMBlock that hands the training problem over to LIBSVM
/** The LIBSVMSolver class implements the Solver interface [see Solver.h] for
 * a SVMBlock [see SVMBlock.h] by handing its training problem over to LIBSVM,
 * the reference implementation of the decomposition method for training a
 * Support Vector Machine. Like SMOSolver, and for the very same reason, it
 * reads the *physical* representation of the SVMBlock: what LIBSVM wants is
 * the data set and the hyper-parameters, and the abstract representation is
 * not needed at all, being only written into by get_var_solution().
 *
 * The interest of this Solver is twofold: it is an independent implementation
 * of the same algorithm SMOSolver implements, hence a reference to check the
 * latter against on any data set, and it is a mature and fast one, hence the
 * natural choice when the data set is large.
 *
 * <b>What LIBSVM can be asked.</b> LIBSVM solves a specific family of
 * training problems, which is smaller than the family a SVMBlock can encode:
 * the assumptions are that
 *
 * - the loss is the linear one, i.e., the slacks are not squared, LIBSVM
 *   having no L2 variant [see SVMBlock::set_squared_loss()];
 *
 * - the bias is not regularised, LIBSVM always having the equality constraint
 *   that the unregularised bias produces [see SVMBlock::set_reg_bias()];
 *
 * - the weight of the regularisation term is 1, LIBSVM having no way to say
 *   otherwise [see SVMBlock::set_reg_weight()]; note that the term could be
 *   scaled away into C, but the value of the training problem would then no
 *   longer be the value LIBSVM reports, and the whole point of this Solver is
 *   to be comparable with the others;
 *
 * - the kernel is one of the four LIBSVM has, i.e., anything but the
 *   Laplacian one [see SVMBlock::set_kernel()].
 *
 * These are checked, and exception is thrown if any of them is not met: an
 * approximation of the problem that has been asked for would be worse than an
 * error, since the value and the model this Solver reports are meant to be
 * compared with those of the Solver that solve the very problem.
 *
 * <b>Re-optimization.</b> There is none: LIBSVM trains from scratch, so every
 * call to compute() re-reads the data set out of the SVMBlock and re-trains.
 * Modification are therefore not queued at all, which also makes this Solver
 * the term of comparison for how much the warm start of SMOSolver is worth.
 *
 * <b>The model.</b> LIBSVM returns the coefficients of the kernel expansion,
 * i.e., the \f$ c_i \f$ of SVMBlock::get_dual_coefficients(), rather than the
 * multipliers; the latter are recovered exactly, since at any optimal
 * solution at most one of the multipliers referring to a sample is nonzero
 * [see the comments to SVMBlock for the dual index space]. Note that the
 * decision function LIBSVM returns is ours and not its opposite only because
 * the targets of a SVCBlock are exactly -1 and +1, which is the one case in
 * which LIBSVM makes sure that +1 is the positive class whatever the order
 * the labels come in; extract_model() checks it rather than assuming it.
 *
 * The value of the Objective is likewise not returned by LIBSVM, and is
 * computed here out of
 * the decision values LIBSVM itself evaluates at the support vectors, which
 * costs a number of kernel evaluations quadratic in *their* number rather
 * than in that of the samples: computing it out of the Gram matrix would
 * defeat the purpose of using LIBSVM in the first place. */

class LIBSVMSolver : public Solver
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*------------------------------ PUBLIC TYPES ------------------------------*/

 using Index = Block::Index;

 using doubleVec = SVMBlock::doubleVec;

/*--------------------- CONSTRUCTOR AND DESTRUCTOR -------------------------*/
/** @name Constructor and destructor
 *  @{ */

 /// constructor of LIBSVMSolver: does nothing special

 LIBSVMSolver( void ) : Solver() {}

/*--------------------------------------------------------------------------*/
 /// destructor of LIBSVMSolver: releases the LIBSVM model, if any

 ~LIBSVMSolver() override { free_model(); }

/** @} ---------------------------------------------------------------------*/
/*--------------------- DERIVED METHODS OF BASE CLASS ----------------------*/
/*--------------------------------------------------------------------------*/
/** @name Public methods derived from base classes
 *  @{ */

 /// sets the Block that the Solver has to solve, which must be a SVMBlock

 void set_Block( Block * block ) override;

/*--------------------------------------------------------------------------*/
 /// LIBSVM cannot re-optimize, hence Modification are not remembered
 /** LIBSVM trains from scratch, and compute() re-reads the data set and the
  * hyper-parameters out of the SVMBlock each time: what has changed in the
  * meanwhile therefore makes no difference, and the Modification is dropped
  * rather than being queued for a re-optimization that cannot happen. */

 void add_Modification( sp_Mod & mod ) override {}

/*--------------------------------------------------------------------------*/
 /// trains the SVM with LIBSVM
 /** Hands the training problem of the SVMBlock over to LIBSVM, which solves
  * it from scratch. Throws exception if the training problem is not one
  * LIBSVM can be asked, see the comments to the class. */

 int compute( bool changedvars = true ) override;

/*--------------------------------------------------------------------------*/
 /// tells whether a solution is available

 bool has_var_solution( void ) override { return( f_solved ); }

/*--------------------------------------------------------------------------*/
 /// writes the multipliers and the bias into the SVMBlock
 /** Writes the model found by the last call to compute() into the SVMBlock,
  * both into its "physical" solution [see SVMBlock::set_dual_solution()] and,
  * if the abstract representation exists, into its Variable. */

 void get_var_solution( Configuration * solc = nullptr ) override;

/*--------------------------------------------------------------------------*/
 /// returns the trained model as a Solution, without going through the Block
 /** Returns the model found by the last call to compute() as a Solution
  * object, or nullptr if there is none. It works exactly as
  * SMOSolver::get_Solution(): if what the Configuration asks for is the
  * model, i.e., a SVMBlockSolution, then it is filled here out of the data
  * of this Solver, without writing into the SVMBlock and hence without
  * requiring any Variable to exist; anything else saves (part of) the
  * abstract representation, which only the SVMBlock can fill, and is
  * therefore left to the method of the base class. */

 [[nodiscard]] Solution * get_Solution( Configuration * solc = nullptr )
  override;

/*--------------------------------------------------------------------------*/
 /// returns a valid lower bound on the optimal objective function value

 OFValue get_lb( void ) override { return( f_value ); }

/*--------------------------------------------------------------------------*/
 /// returns a valid upper bound on the optimal objective function value

 OFValue get_ub( void ) override { return( f_value ); }

/*--------------------------------------------------------------------------*/
 /// returns the value of the current solution, if any

 OFValue get_var_value( void ) override { return( f_value ); }

/*--------------------------------------------------------------------------*/
 /// extends Solver::dbl_par_type_S with the LIBSVMSolver parameters

 enum dbl_par_type_LSVM {
  dblLSVMTol = dblLastAlgPar ,  ///< tolerance on the optimality conditions
  /**< The tolerance LIBSVM stops at, i.e., the maximal violation of the
   * optimality conditions of the dual it tolerates; it is the "eps" of
   * LIBSVM, and it defaults to 1e-3 exactly as there. */
  dblLSVMCache ,                ///< size of the kernel cache, in MB
  /**< The size, in MB, of the cache LIBSVM keeps the columns of the Hessian
   * it has already computed into. Defaults to 100, as in LIBSVM. */
  dblLastAlgParLSVM  ///< 1st allowed new double parameter for derived classes
  };

/*--------------------------------------------------------------------------*/
 /// extends Solver::int_par_type_S with the LIBSVMSolver parameters

 enum int_par_type_LSVM {
  intLSVMShrink = intLastAlgPar ,  ///< nonzero to use the shrinking heuristic
  /**< Nonzero if LIBSVM has to use its shrinking heuristic, which drops from
   * the working set the multipliers that are unlikely to move again.
   * Defaults to 1, as in LIBSVM. */
  intLastAlgParLSVM  ///< 1st allowed new int parameter for derived classes
  };

 using Solver::set_par;  // keep the other set_par() overloads visible

 /// honoured parameters: dblLSVMTol, dblLSVMCache
 void set_par( idx_type par , double value ) override {
  if( par == dblLSVMTol ) { f_tol = value; return; }
  if( par == dblLSVMCache ) { f_cache = value; return; }
  Solver::set_par( par , value );
  }

 /// honoured parameters: intLSVMShrink, intLogVerb
 void set_par( idx_type par , int value ) override {
  if( par == intLSVMShrink ) { f_shrink = value; return; }
  if( par == intLogVerb ) { f_log_verb = value; return; }
  Solver::set_par( par , value );
  }

 [[nodiscard]] double get_dbl_par( idx_type par ) const override {
  if( par == dblLSVMTol ) return( f_tol );
  if( par == dblLSVMCache ) return( f_cache );
  return( Solver::get_dbl_par( par ) );
  }

 [[nodiscard]] int get_int_par( idx_type par ) const override {
  if( par == intLSVMShrink ) return( f_shrink );
  if( par == intLogVerb ) return( f_log_verb );
  return( Solver::get_int_par( par ) );
  }

 [[nodiscard]] idx_type get_num_dbl_par( void ) const override {
  return( Solver::get_num_dbl_par() + dblLastAlgParLSVM - dblLastAlgPar );
  }

 [[nodiscard]] idx_type get_num_int_par( void ) const override {
  return( Solver::get_num_int_par() + intLastAlgParLSVM - intLastAlgPar );
  }

 [[nodiscard]] double get_dflt_dbl_par( idx_type par ) const override {
  if( par == dblLSVMTol ) return( 1e-3 );
  if( par == dblLSVMCache ) return( 100 );
  return( Solver::get_dflt_dbl_par( par ) );
  }

 [[nodiscard]] int get_dflt_int_par( idx_type par ) const override {
  if( par == intLSVMShrink ) return( 1 );
  return( Solver::get_dflt_int_par( par ) );
  }

 [[nodiscard]] idx_type dbl_par_str2idx( const std::string & name )
  const override {
  if( name == "dblLSVMTol" ) return( dblLSVMTol );
  if( name == "dblLSVMCache" ) return( dblLSVMCache );
  return( Solver::dbl_par_str2idx( name ) );
  }

 [[nodiscard]] idx_type int_par_str2idx( const std::string & name )
  const override {
  if( name == "intLSVMShrink" ) return( intLSVMShrink );
  return( Solver::int_par_str2idx( name ) );
  }

 [[nodiscard]] const std::string & dbl_par_idx2str( idx_type idx )
  const override {
  static const std::string tol = "dblLSVMTol";
  static const std::string cache = "dblLSVMCache";
  if( idx == dblLSVMTol ) return( tol );
  if( idx == dblLSVMCache ) return( cache );
  return( Solver::dbl_par_idx2str( idx ) );
  }

 [[nodiscard]] const std::string & int_par_idx2str( idx_type idx )
  const override {
  static const std::string shrink = "intLSVMShrink";
  return( idx == intLSVMShrink ? shrink : Solver::int_par_idx2str( idx ) );
  }

/** @} ---------------------------------------------------------------------*/
/*--------------------- OTHER METHODS OF THE CLASS -------------------------*/
/*--------------------------------------------------------------------------*/

 /// returns the number of support vectors of the trained model

 Index get_NSV( void ) const { return( f_model ? f_model->l : 0 ); }

/*---------------------- PROTECTED PART OF THE CLASS -----------------------*/

 protected:

/*--------------------------- PROTECTED METHODS ----------------------------*/

 /// throws if the training problem is not one LIBSVM can be asked

 void check_supported( void ) const;

/*--------------------------------------------------------------------------*/
 /// fills the LIBSVM problem out of the data set of the SVMBlock

 void build_problem( void );

/*--------------------------------------------------------------------------*/
 /// fills the LIBSVM parameters out of the hyper-parameters of the SVMBlock

 void build_parameters( void );

/*--------------------------------------------------------------------------*/
 /// recovers the multipliers, the bias and the value out of the LIBSVM model

 void extract_model( void );

/*--------------------------------------------------------------------------*/
 /// releases the LIBSVM model, if any

 void free_model( void ) {
  if( f_model )
   svm_free_and_destroy_model( & f_model );
  }

/*---------------------------- PROTECTED FIELDS ----------------------------*/

 SVMBlock * f_SVM = nullptr;   ///< the SVMBlock to be solved

 double f_tol = 1e-3;          ///< tolerance on the optimality conditions
 double f_cache = 100;         ///< size of the kernel cache, in MB
 int f_shrink = 1;             ///< nonzero to use the shrinking heuristic
 int f_log_verb = 0;           ///< nonzero for LIBSVM to print its log

 bool f_solved = false;        ///< true if a solution is available
 double f_value = 0;           ///< the value of the dual at the solution
 double f_b = 0;               ///< the bias of the model

 doubleVec v_alpha;            ///< the multipliers of the model

 // the problem as LIBSVM wants it- - - - - - - - - - - - - - - - - - - - - -

 svm_problem f_prob = {};      ///< the data set
 svm_parameter f_par = {};     ///< the hyper-parameters
 svm_model * f_model = nullptr;    ///< the trained model

 /* The samples are handed over as the sparse rows LIBSVM wants, which have
  * to stay alive as long as the model does: LIBSVM does not copy them, the
  * support vectors of the model being pointers into them. */

 std::vector< svm_node > v_node;      ///< the samples, sparse and in a row
 std::vector< svm_node * > v_row;     ///< where each sample starts
 doubleVec v_target;                  ///< the targets, as LIBSVM wants them

/*----------------------- PRIVATE PART OF THE CLASS ------------------------*/

 private:

/*---------------------------- PRIVATE METHODS -----------------------------*/

 SMSpp_insert_in_factory_h;  // insert LIBSVMSolver in the Solver factory

/*--------------------------------------------------------------------------*/

 };  // end( class( LIBSVMSolver ) )

/** @} end( group( LIBSVMSolver_CLASSES ) ) */

/*--------------------------------------------------------------------------*/

 }  // end( namespace SMSpp_di_unipi_it )

/*--------------------------------------------------------------------------*/

#endif  /* LIBSVMSolver.h included */

/*--------------------------------------------------------------------------*/
/*------------------------ End File LIBSVMSolver.h -------------------------*/
/*--------------------------------------------------------------------------*/
