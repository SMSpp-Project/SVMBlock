/*--------------------------------------------------------------------------*/
/*--------------------------- File SMOSolver.h -----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Header file for the *concrete* class SMOSolver, which implements the Solver
 * interface for a SVMBlock with the Sequential Minimal Optimization algorithm
 * on the dual formulation of the training problem.
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

#ifndef __SMOSolver
 #define __SMOSolver
                      /* self-identification: #endif at the end of the file */

/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include "SVMBlock.h"

#include "Solver.h"

/*--------------------------------------------------------------------------*/
/*------------------------------ NAMESPACE ---------------------------------*/
/*--------------------------------------------------------------------------*/

/// namespace for the Structured Modeling System++ (SMS++)
namespace SMSpp_di_unipi_it
{
/*--------------------------------------------------------------------------*/
/*-------------------------------- CLASSES ---------------------------------*/
/*--------------------------------------------------------------------------*/
/** @defgroup SMOSolver_CLASSES Classes in SMOSolver.h
 *  @{ */

/*--------------------------------------------------------------------------*/
/*---------------------------- CLASS SMOSolver -----------------------------*/
/*--------------------------------------------------------------------------*/
/*--------------------------- GENERAL NOTES --------------------------------*/
/*--------------------------------------------------------------------------*/
/// a Solver for a SVMBlock based on Sequential Minimal Optimization
/** The SMOSolver class implements the Solver interface [see Solver.h] for a
 * SVMBlock [see SVMBlock.h], solving the dual formulation of the training
 * problem with the Sequential Minimal Optimization (SMO) algorithm, i.e., the
 * decomposition method that at each iteration optimizes over the smallest
 * possible working set and therefore never needs the Hessian of the dual as a
 * whole. Since the Hessian is dense, this is what makes the method the
 * standard choice for training a SVM, and it is the reason why the Solver
 * reads the data out of the *physical* representation of the SVMBlock: the
 * abstract representation is not needed and, if it exists, it is only written
 * into by get_var_solution().
 *
 * The dual is
 * \f[
 *   \min_{ \alpha } \quad \frac{1}{2} \alpha^T Q \alpha + q^T \alpha
 *   \quad , \quad 0 \leq \alpha_k \leq u \quad , \quad
 *   [ \; s^T \alpha = 0 \; ]
 * \f]
 * see the comments to SVMBlock for the meaning of the data. Two cases have to
 * be distinguished, according to whether the equality constraint is there:
 *
 * - if it is, at each iteration the *maximal violating pair* \f$ ( i , j ) \f$
 *   of the optimality conditions is selected and the objective is minimized
 *   exactly along the only feasible direction that changes those two
 *   multipliers alone, i.e., \f$ \alpha_i \mathrel{+}= s_i t \f$ and
 *   \f$ \alpha_j \mathrel{-}= s_j t \f$, which is the SMO step proper;
 *
 * - if the bias is regularised there is no equality constraint, hence a single
 *   multiplier at a time can be moved, and the method degenerates into the
 *   (greedily selected) coordinate descent on a box-constrained problem.
 *
 * In both cases the algorithm maintains the gradient of the dual and stops
 * when the maximal violation of the optimality conditions falls below
 * dblSMOTol. Because the dual is a convex quadratic program, the value at
 * termination is (within that tolerance) the optimal one, and it is reported
 * as both the lower and the upper bound. What is reported is the value of the
 * Objective of the SVMBlock, hence the value of the dual only if that is the
 * formulation its abstract representation encodes: for the primal ones the
 * opposite is reported, strong duality holding since the training problem is
 * convex. This is what makes the Solver directly comparable with any other
 * one attached to the same SVMBlock.
 *
 * The bias of the model is recovered as the multiplier of the equality
 * constraint, i.e., as the midpoint of the interval that the optimality
 * conditions leave for it, and written into the SVMBlock together with the
 * multipliers by get_var_solution().
 *
 * The two-multiplier step is the one of
 *
 * J. C. Platt "Sequential Minimal Optimization: A Fast Algorithm for Training
 * Support Vector Machines" *Microsoft Research technical report* MSR-TR-98-14,
 * 1998
 *
 * while the working set is selected, and the algorithm is stopped, with the
 * two thresholds of
 *
 * S. S. Keerthi, S. K. Shevade, C. Bhattacharyya, K. R. K. Murthy
 * "Improvements to Platt's SMO Algorithm for SVM Classifier Design"
 * *Neural Computation* 13(3), 637-649, 2001
 *
 * which for the regression case reduce to the thresholds of
 *
 * S. K. Shevade, S. S. Keerthi, C. Bhattacharyya, K. R. K. Murthy
 * "Improvements to the SMO Algorithm for SVM Regression" *IEEE Transactions
 * on Neural Networks* 11(5), 1188-1193, 2000
 *
 * since a regression sample contributes two dual indices with opposite signs,
 * whence the two sides of its insensitivity tube.
 *
 * <b>Re-optimization.</b> The Solver keeps the multipliers and the gradient of
 * the dual at them across the calls to compute(), and reads the Modification
 * that the SVMBlock issues [see SVMBlockMod] to find out what it has to do
 * with them. Whatever changes the Hessian of the dual, i.e., the kernel, the
 * data set or the regularisation of the bias, leaves nothing to be re-used and
 * the algorithm restarts from the origin; anything else, i.e., the trade-off
 * parameter, the shape of the loss, the half-width of the insensitivity tube
 * and the targets of a regression problem, only changes the bounds, the linear
 * term or the diagonal, and therefore the previous multipliers are still a
 * sensible starting point:
 *
 * - if the upper bound has decreased, the multipliers are *scaled* rather than
 *   clipped, which is what keeps them feasible for the equality constraint as
 *   well, since the latter is homogeneous;
 *
 * - the gradient is then updated in \f$ O( N ) \f$ time, being affine in the
 *   multipliers, in the linear term and in the diagonal alike, and therefore
 *   it is exactly the gradient at the new point of the new dual, as if it had
 *   been recomputed from scratch in \f$ O( N^2 ) \f$ time.
 *
 * This is what makes a model selection, where the very same data set is
 * trained over and over with different hyper-parameters, cost much less than
 * the sum of the individual trainings, and it costs nothing when nothing has
 * changed, in which case compute() returns immediately. */

class SMOSolver : public Solver
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*------------------------------ PUBLIC TYPES ------------------------------*/

 using Index = Block::Index;

 using doubleVec = SVMBlock::doubleVec;

/*--------------------- CONSTRUCTOR AND DESTRUCTOR -------------------------*/
/** @name Constructor and destructor
 *  @{ */

 /// constructor of SMOSolver: does nothing special

 SMOSolver( void ) : Solver() {}

/*--------------------------------------------------------------------------*/
 /// destructor of SMOSolver: does nothing special

 ~SMOSolver() override = default;

/** @} ---------------------------------------------------------------------*/
/*--------------------- DERIVED METHODS OF BASE CLASS ----------------------*/
/*--------------------------------------------------------------------------*/
/** @name Public methods derived from base classes
 *  @{ */

 /// sets the Block that the Solver has to solve, which must be a SVMBlock

 void set_Block( Block * block ) override;

/*--------------------------------------------------------------------------*/
 /// solves the dual of the training problem
 /** Solves the dual of the training problem, starting from the solution of
  * the previous call if the Modification issued by the SVMBlock in the
  * meantime allow it, and from the origin otherwise; see the comments to the
  * class for the details. */

 int compute( bool changedvars = true ) override;

/*--------------------------------------------------------------------------*/
 /// tells whether a solution is available

 bool has_var_solution( void ) override { return( f_solved ); }

/*--------------------------------------------------------------------------*/
 /// writes the multipliers and the bias into the SVMBlock
 /** Writes the multipliers and the bias found by the last call to compute()
  * into the SVMBlock, both into its "physical" solution [see
  * SVMBlock::set_dual_solution()] and, if the abstract representation of the
  * dual formulation exists, into its Variable. */

 void get_var_solution( Configuration * solc = nullptr ) override;

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
 /// extends Solver::dbl_par_type_S with the SMOSolver parameters

 enum dbl_par_type_SMOS {
  dblSMOTol = dblLastAlgPar ,  ///< tolerance on the optimality conditions
  /**< The algorithm stops as soon as the maximal violation of the optimality
   * conditions of the dual, i.e., the gap between the largest value that the
   * bias may take according to the multipliers that can be increased and the
   * smallest one according to those that can be decreased, falls below this
   * value. Defaults to 1e-3. */
  dblLastAlgParSMOS  ///< 1st allowed new double parameter for derived classes
  };

 using Solver::set_par;  // keep the other set_par() overloads visible

 /// honoured parameter: dblSMOTol
 void set_par( idx_type par , double value ) override {
  if( par == dblSMOTol ) { f_tol = value; return; }
  Solver::set_par( par , value );
  }

 /// honoured parameter: intMaxIter
 void set_par( idx_type par , int value ) override {
  if( par == intMaxIter ) { f_max_iter = value; return; }
  Solver::set_par( par , value );
  }

 [[nodiscard]] double get_dbl_par( idx_type par ) const override {
  if( par == dblSMOTol ) return( f_tol );
  return( Solver::get_dbl_par( par ) );
  }

 [[nodiscard]] int get_int_par( idx_type par ) const override {
  if( par == intMaxIter ) return( f_max_iter );
  return( Solver::get_int_par( par ) );
  }

 [[nodiscard]] idx_type get_num_dbl_par( void ) const override {
  return( Solver::get_num_dbl_par() + dblLastAlgParSMOS - dblLastAlgPar );
  }

 [[nodiscard]] double get_dflt_dbl_par( idx_type par ) const override {
  return( par == dblSMOTol ? 1e-3 : Solver::get_dflt_dbl_par( par ) );
  }

 [[nodiscard]] idx_type dbl_par_str2idx( const std::string & name )
  const override {
  return( name == "dblSMOTol" ? dblSMOTol : Solver::dbl_par_str2idx( name ) );
  }

 [[nodiscard]] const std::string & dbl_par_idx2str( idx_type idx )
  const override {
  static const std::string name = "dblSMOTol";
  return( idx == dblSMOTol ? name : Solver::dbl_par_idx2str( idx ) );
  }

/** @} ---------------------------------------------------------------------*/
/*--------------------- OTHER METHODS OF THE CLASS -------------------------*/
/*--------------------------------------------------------------------------*/

 /// returns the number of iterations of the last call to compute()

 Index get_iter( void ) const { return( f_iter ); }

/*---------------------- PROTECTED PART OF THE CLASS -----------------------*/

 protected:

/*--------------------------- PROTECTED METHODS ----------------------------*/

 /// tells whether the given Modification allows a warm start
 /** Returns true if the given Modification, and recursively all those a
  * GroupModification contains, leave the multipliers of the previous call to
  * compute() worth starting from, and false if the algorithm rather has to
  * restart from the origin. */

 bool guts_of_poM( const Modification * mod ) const;

/*--------------------------------------------------------------------------*/
 /// reads the data of the dual out of the SVMBlock, starting from the origin

 void reload( void );

/*--------------------------------------------------------------------------*/
 /// realigns the cached data, the multipliers and the gradient to the SVMBlock
 /** Realigns the data of the dual cached out of the SVMBlock to the current
  * ones, and with them the multipliers of the previous call to compute() and
  * the gradient of the dual at them, so that the algorithm can restart from
  * there. Returns false if the change turns out not to be one it can follow,
  * in which case the caller has to reload() everything. */

 bool resync( void );

/*--------------------------------------------------------------------------*/
 /// the SMO iteration proper, for the dual with the equality constraint

 int solve_with_equality( void );

/*--------------------------------------------------------------------------*/
 /// the coordinate descent, for the dual without the equality constraint

 int solve_box( void );

/*--------------------------------------------------------------------------*/
 /// returns the entry ( k , l ) of the Hessian of the dual

 double Q( Index k , Index l ) const
 {
  double q = f_ds[ k ] * f_ds[ l ] *
             ( f_K[ std::size_t( f_di[ k ] ) * f_n + f_di[ l ] ] + f_rb );
  if( k == l )
   q += f_d;
  return( q );
  }

/*---------------------------- PROTECTED FIELDS ----------------------------*/

 SVMBlock * f_SVM = nullptr;   ///< the SVMBlock to be solved

 double f_tol = 1e-3;          ///< tolerance on the optimality conditions
 int f_max_iter = -1;          ///< maximum number of iterations, < 0 = none

 bool f_solved = false;        ///< true if a solution is available
 double f_value = 0;           ///< the value of the dual at the solution
 double f_b = 0;               ///< the bias of the model
 Index f_iter = 0;             ///< iterations of the last call to compute()

 doubleVec v_alpha;            ///< the current multipliers
 doubleVec v_G;                ///< the gradient of the dual at them

 // the data of the dual, cached out of the SVMBlock- - - - - - - - - - - - -

 Index f_n = 0;                ///< the number n of samples
 Index f_N = 0;                ///< the size N of the dual index space
 double f_u = 0;               ///< the upper bound on the multipliers
 double f_rb = 0;              ///< 1 if the bias is regularised, 0 otherwise
 double f_d = 0;               ///< the diagonal term due to the squared loss
 const double * f_K = nullptr;      ///< the n x n Gram matrix
 const Index * f_di = nullptr;      ///< the N sample indices

 /* The signs and the linear coefficients are *copied* rather than pointed to
  * in the SVMBlock, since the gradient is updated with the difference between
  * their new and their old value, which requires having the latter around
  * once the SVMBlock has changed. */

 doubleVec v_s;                ///< the N signs
 doubleVec v_q;                ///< the N linear coefficients

 const double * f_ds = nullptr;     ///< shortcut to v_s.data()
 const double * f_dq = nullptr;     ///< shortcut to v_q.data()

/*----------------------- PRIVATE PART OF THE CLASS ------------------------*/

 private:

/*---------------------------- PRIVATE METHODS -----------------------------*/

 SMSpp_insert_in_factory_h;  // insert SMOSolver in the Solver factory

/*--------------------------------------------------------------------------*/

 };  // end( class( SMOSolver ) )

/** @} end( group( SMOSolver_CLASSES ) ) */

/*--------------------------------------------------------------------------*/

 }  // end( namespace SMSpp_di_unipi_it )

/*--------------------------------------------------------------------------*/

#endif  /* SMOSolver.h included */

/*--------------------------------------------------------------------------*/
/*------------------------- End File SMOSolver.h ---------------------------*/
/*--------------------------------------------------------------------------*/
