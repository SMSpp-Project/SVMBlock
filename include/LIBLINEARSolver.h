/*--------------------------------------------------------------------------*/
/*------------------------ File LIBLINEARSolver.h --------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Header file for the *concrete* class LIBLINEARSolver, which implements the
 * Solver interface for a SVMBlock by handing the training problem over to
 * LIBLINEAR.
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

#ifndef __LIBLINEARSolver
 #define __LIBLINEARSolver
                      /* self-identification: #endif at the end of the file */

/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include "SVMBlock.h"

#include "Solver.h"

#include <linear.h>

/*--------------------------------------------------------------------------*/
/*------------------------------ NAMESPACE ---------------------------------*/
/*--------------------------------------------------------------------------*/

/// namespace for the Structured Modeling System++ (SMS++)
namespace SMSpp_di_unipi_it
{
/*--------------------------------------------------------------------------*/
/*-------------------------------- CLASSES ---------------------------------*/
/*--------------------------------------------------------------------------*/
/** @defgroup LIBLINEARSolver_CLASSES Classes in LIBLINEARSolver.h
 *  @{ */

/*--------------------------------------------------------------------------*/
/*-------------------------- CLASS LIBLINEARSolver -------------------------*/
/*--------------------------------------------------------------------------*/
/*--------------------------- GENERAL NOTES --------------------------------*/
/*--------------------------------------------------------------------------*/
/// a Solver for a SVMBlock that hands the training problem over to LIBLINEAR
/** The LIBLINEARSolver class implements the Solver interface [see Solver.h]
 * for a SVMBlock [see SVMBlock.h] by handing its training problem over to
 * LIBLINEAR, which trains a linear model by working on the primal, or on its
 * dual, in the \f$ m \f$ weights rather than in the \f$ N \f$ multipliers.
 * Like SMOSolver and LIBSVMSolver, and for the very same reason, it reads the
 * *physical* representation of the SVMBlock: what LIBLINEAR wants is the data
 * set and the hyper-parameters.
 *
 * This Solver and LIBSVMSolver cover the two sides of the same question, and
 * neither covers both:
 *
 * - LIBSVM has any kernel, but only the linear loss, and its bias is always
 *   the unregularised one, i.e., the equality constraint is always there;
 *
 * - LIBLINEAR has the linear kernel only, but both losses, and its bias is
 *   the regularised one, being one more component of the model: it is
 *   appended to each sample as a further feature whose value is one.
 *
 * What LIBLINEAR solves is therefore \f$ \min \frac{1}{2} \| w \|^2 +
 * C' \sum_i \ell( \xi_i ) \f$, which is the training problem
 * \f$ \min \frac{\rho}{2} ( \| w \|^2 + b^2 ) + C \sum_i \ell( \xi_i ) \f$
 * of the SVMBlock up to the scaling \f$ \rho \f$ of the regularisation term:
 * dividing the objective by \f$ \rho \f$ leaves the same minimiser, hence
 * \f$ C' = C / \rho \f$ for either loss, the two writing the loss in the same
 * way, and the value that this Solver reports is the one of the SVMBlock,
 * computed out of the model.
 *
 * A training problem this Solver is *not* able to take is refused by
 * throwing, rather than solved as a different one: a nonlinear kernel, an
 * unregularised bias, and the linear term that the subproblem of a chunk
 * carries [see SVMBlock::has_linear_term()].
 *
 * The tolerance is honoured, but the number of iterations of LIBLINEAR is
 * capped, and the cap is written in its sources rather than being a
 * parameter: the dual solvers stop after 300 iterations whatever the
 * tolerance asks. A model that comes out of a capped run is a feasible one,
 * but it is not the optimal one, and the value that goes with it is an upper
 * bound on the optimal value and nothing more, so compute() reports it as
 * kLowPrecision. The alternative, i.e., reporting kOK, would make a
 * comparison against this Solver read as if the two had solved the same
 * problem to the same accuracy. */

class LIBLINEARSolver : public Solver
{

/*--------------------------------------------------------------------------*/
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/
/*--------------------------------------------------------------------------*/

 public:

/*--------------------------------------------------------------------------*/
/*---------------------------- PUBLIC TYPES --------------------------------*/
/*--------------------------------------------------------------------------*/

 using Index = Block::Index;

 using doubleVec = SVMBlock::doubleVec;

/*--------------------------------------------------------------------------*/
/*---------------------- CONSTRUCTOR AND DESTRUCTOR ------------------------*/
/*--------------------------------------------------------------------------*/
/** @name Constructor and destructor
 *  @{ */

 /// constructor: does nothing special

 LIBLINEARSolver( void ) : Solver() {}

/*--------------------------------------------------------------------------*/
 /// destructor: releases the LIBLINEAR model

 ~LIBLINEARSolver() override { free_model(); }

/** @} ---------------------------------------------------------------------*/
/*-------------------- DERIVED METHODS OF BASE CLASS -----------------------*/
/*--------------------------------------------------------------------------*/
/** @name Derived methods of the base class
 *  @{ */

 /// sets the SVMBlock this Solver is attached to

 void set_Block( Block * block ) override;

/*--------------------------------------------------------------------------*/
 /// LIBLINEAR cannot re-optimize, hence a Modification only says "train
 /// again"
 /** LIBLINEAR trains from scratch, and compute() re-reads the data set and
  * the hyper-parameters out of the SVMBlock each time: what has changed in
  * the meanwhile therefore makes no difference, and the Modification is not
  * kept, only the fact that there was one. */

 void add_Modification( sp_Mod & mod ) override { f_dirty = true; }

/*--------------------------------------------------------------------------*/
 /// trains the SVM with LIBLINEAR
 /** Hands the training problem of the SVMBlock over to LIBLINEAR, which
  * solves it from scratch; it returns at once if nothing has changed since
  * the previous call. Returns kOK, or kLowPrecision if LIBLINEAR has stopped
  * on its own cap on the number of iterations. Throws exception if the
  * training problem is not one LIBLINEAR can be asked, see the comments to
  * the class. */

 int compute( bool changedvars = true ) override;

/*--------------------------------------------------------------------------*/
 /// tells whether a solution is available

 bool has_var_solution( void ) override { return( f_solved ); }

/*--------------------------------------------------------------------------*/
 /// writes the weights and the bias into the SVMBlock
 /** Writes the model found by the last call to compute() into the SVMBlock,
  * both into its "physical" solution [see SVMBlock::set_primal_solution()]
  * and, if the abstract representation exists, into its Variable. Since what
  * LIBLINEAR returns is the model and not the multipliers, the latter are
  * left at zero, exactly as they are when the model comes out of a primal. */

 void get_var_solution( Configuration * solc = nullptr ) override;

/*--------------------------------------------------------------------------*/
 /// returns the model found by the last call to compute() as a Solution

 Solution * get_Solution( Configuration * solc = nullptr ) override;

/*--------------------------------------------------------------------------*/
 /// returns a valid lower bound on the optimal objective function value
 /** The value of the model LIBLINEAR has found, which is a valid lower bound
  * only when it is the optimal one, i.e., when the run has not stopped on
  * the cap on the number of iterations; -Inf is returned if it has. */

 OFValue get_lb( void ) override {
  return( f_capped ? - Inf< OFValue >() : f_value );
  }

/*--------------------------------------------------------------------------*/
 /// returns a valid upper bound on the optimal objective function value
 /** The value of the model LIBLINEAR has found: the training problem being a
  * minimisation, the value of any model is an upper bound on the optimal
  * one, whether the run has been capped or not. */

 OFValue get_ub( void ) override { return( f_value ); }

/*--------------------------------------------------------------------------*/
 /// returns the value of the current solution, if any

 OFValue get_var_value( void ) override { return( f_value ); }

/*--------------------------------------------------------------------------*/
 /// extends Solver::dbl_par_type_S with the LIBLINEARSolver parameters

 enum dbl_par_type_LLIN {
  dblLLINTol = dblLastAlgPar ,  ///< tolerance on the optimality conditions
  /**< The tolerance LIBLINEAR stops at, i.e., its "eps"; it defaults to
   * 1e-3, which is tighter than the default of LIBLINEAR itself, that
   * depends on the solver type and is as loose as 1e-1 for the dual ones.
   * The cap on the number of iterations may well be reached before the
   * tolerance is, in which case compute() reports kLowPrecision. */
  dblLastAlgParLLIN  ///< 1st allowed new double parameter for derived classes
  };

 using Solver::set_par;  // keep the other set_par() overloads visible

 /// honoured parameters: dblLLINTol
 void set_par( idx_type par , double value ) override {
  if( par == dblLLINTol ) { f_tol = value; return; }
  Solver::set_par( par , value );
  }

 /// honoured parameters: intLogVerb
 void set_par( idx_type par , int value ) override {
  if( par == intLogVerb ) { f_log_verb = value; return; }
  Solver::set_par( par , value );
  }

 [[nodiscard]] double get_dbl_par( idx_type par ) const override {
  if( par == dblLLINTol ) return( f_tol );
  return( Solver::get_dbl_par( par ) );
  }

 [[nodiscard]] int get_int_par( idx_type par ) const override {
  if( par == intLogVerb ) return( f_log_verb );
  return( Solver::get_int_par( par ) );
  }

 [[nodiscard]] idx_type get_num_dbl_par( void ) const override {
  return( Solver::get_num_dbl_par() + dblLastAlgParLLIN - dblLastAlgPar );
  }

 [[nodiscard]] double get_dflt_dbl_par( idx_type par ) const override {
  if( par == dblLLINTol ) return( 1e-3 );
  return( Solver::get_dflt_dbl_par( par ) );
  }

 [[nodiscard]] idx_type dbl_par_str2idx( const std::string & name )
  const override {
  if( name == "dblLLINTol" ) return( dblLLINTol );
  return( Solver::dbl_par_str2idx( name ) );
  }

 [[nodiscard]] const std::string & dbl_par_idx2str( idx_type idx )
  const override {
  static const std::string tol = "dblLLINTol";
  return( idx == dblLLINTol ? tol : Solver::dbl_par_idx2str( idx ) );
  }

/** @} ---------------------------------------------------------------------*/
/*--------------------- OTHER METHODS OF THE CLASS -------------------------*/
/*--------------------------------------------------------------------------*/

 /// tells whether LIBLINEAR has stopped on its cap on the iterations

 bool is_capped( void ) const { return( f_capped ); }

/*---------------------- PROTECTED PART OF THE CLASS -----------------------*/

 protected:

/*--------------------------- PROTECTED METHODS ----------------------------*/

 /// throws if the training problem is not one LIBLINEAR can be asked

 void check_supported( void ) const;

/*--------------------------------------------------------------------------*/
 /// fills the LIBLINEAR problem out of the data set of the SVMBlock

 void build_problem( void );

/*--------------------------------------------------------------------------*/
 /// fills the LIBLINEAR parameters out of the hyper-parameters of the
 /// SVMBlock

 void build_parameters( void );

/*--------------------------------------------------------------------------*/
 /// recovers the weights, the bias and the value out of the LIBLINEAR model

 void extract_model( void );

/*--------------------------------------------------------------------------*/
 /// releases the LIBLINEAR model, if any
 /** Releases the model and forgets it. The pointer is zeroed here rather
  * than being left to LIBLINEAR: free_and_destroy_model() only zeroes it
  * from 2.40 on, so on an older one the model would be released twice, once
  * by the next compute() and once by the destructor. */

 void free_model( void ) {
  if( f_model ) {
   free_and_destroy_model( & f_model );
   f_model = nullptr;
   }
  }

/*---------------------------- PROTECTED FIELDS ----------------------------*/

 SVMBlock * f_SVM = nullptr;   ///< the SVMBlock to be solved

 double f_tol = 1e-3;          ///< tolerance on the optimality conditions
 int f_log_verb = 0;           ///< nonzero for LIBLINEAR to print its log

 bool f_solved = false;        ///< true if a solution is available
 bool f_dirty = true;          ///< true if the SVMBlock has changed since
 bool f_capped = false;        ///< true if the iterations have been capped
 double f_value = 0;           ///< the value of the training problem
 double f_b = 0;               ///< the bias of the model

 doubleVec v_w;                ///< the weights of the model

 // the problem as LIBLINEAR wants it - - - - - - - - - - - - - - - - - - - -

 problem f_prob = {};          ///< the data set
 parameter f_par = {};         ///< the hyper-parameters
 model * f_model = nullptr;    ///< the trained model

 /* The samples are handed over as the sparse rows LIBLINEAR wants, which
  * have to stay alive as long as train() runs, LIBLINEAR not copying them. */

 std::vector< feature_node > v_node;   ///< the samples, sparse and in a row
 std::vector< feature_node * > v_row;  ///< where each sample starts
 doubleVec v_target;                   ///< the targets, as LIBLINEAR wants

/*----------------------- PRIVATE PART OF THE CLASS ------------------------*/

 private:

/*---------------------------- PRIVATE METHODS -----------------------------*/

 SMSpp_insert_in_factory_h;  // insert LIBLINEARSolver in the Solver factory

/*--------------------------------------------------------------------------*/

 };  // end( class( LIBLINEARSolver ) )

/** @} end( group( LIBLINEARSolver_CLASSES ) ) */

/*--------------------------------------------------------------------------*/

 }  // end( namespace SMSpp_di_unipi_it )

/*--------------------------------------------------------------------------*/

#endif  /* LIBLINEARSolver.h included */

/*--------------------------------------------------------------------------*/
/*----------------------- End File LIBLINEARSolver.h -----------------------*/
/*--------------------------------------------------------------------------*/
