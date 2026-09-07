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
 * changed, in which case compute() returns immediately.
 *
 * <b>The Lagrangian subproblem of a chunk.</b> The linear term of the primal
 * [see SVMBlock::set_linear_term()], i.e., what the Lagrangian relaxation of
 * the constraints linking a chunk of the consensus structure to the others
 * leaves in the subproblem of that chunk, is dealt with exactly as anything
 * else that only changes the linear term of the dual: it shifts \f$ q \f$
 * and the right-hand side of the equality constraint, and it adds a constant
 * to the value. Because the shifts are affine in the multipliers as well, a
 * change of the multipliers of the linking constraints, i.e., what a
 * Lagrangian Solver does at each of its iterations, is followed in
 * \f$ O( N ) \f$ time from the previous solution: this is what makes the
 * Solver usable inside a Lagrangian scheme, where the very same chunk is
 * solved over and over with different multipliers. The multipliers are then
 * moved back onto the equality constraint, whose right-hand side has changed,
 * one coordinate at a time; if the bounds leave no room for this the dual is
 * empty, i.e., the subproblem is unbounded, and kUnbounded is reported. */

class SMOSolver : public Solver
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*------------------------------ PUBLIC TYPES ------------------------------*/

 using Index = Block::Index;

 using doubleVec = SVMBlock::doubleVec;

 using IndexVec = SVMBlock::IndexVec;

 using Subset = Block::Subset;

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
 /// returns the trained model as a Solution, without going through the Block
 /** Returns the solution of the last call to compute() as a Solution object,
  * or nullptr if there is none.
  *
  * If what the Configuration asks for is a SVMBlockSolution, i.e., the
  * trained model [see SVMBlockSolution], then it is built here out of the
  * multipliers this Solver holds, without writing anything into the SVMBlock:
  * no Variable is therefore required to exist, and the SVMBlock is not
  * lock()-ed, so that any number of Solver attached to it can produce their
  * own Solution at the same time.
  *
  * Any other Solution saves (part of) the abstract representation instead,
  * which only the SVMBlock can fill; the method of the base class is used
  * then, which writes the solution into the Variable and asks the SVMBlock
  * for it. Which one is asked for is not decided here: the SVMBlock is asked
  * for an empty Solution, exactly as it would be in the end, and what it
  * returns is what tells the two cases apart. */

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

/*--------------------------------------------------------------------------*/
 /// extends Solver::int_par_type_S with the SMOSolver parameters

 enum int_par_type_SMOS {
  intSMOPath = intLastAlgPar ,  ///< whether new samples are learnt exactly
  /**< Nonzero if the samples that have been *added* to the SVMBlock since
   * the previous call are learnt by following the exact solution path [see
   * follow_path()] rather than by iterating from the previous solution: the
   * path leaves the solution optimal at every step, hence when the last
   * added sample has been learnt there is nothing left to iterate on.
   * Defaults to 1. Note that the path is only taken when nothing else has
   * changed and the previous call did find an optimal solution; whatever the
   * value of this parameter, the conditions are checked and the iteration
   * finishes the job if they do not hold. */

  intSMOShrink ,  ///< whether the active set is shrunk
  /**< Nonzero if the multipliers that provably cannot be selected are taken
   * out of the active set, so that the iterations, which cost \f$ O( | A | )
   * \f$, get cheaper as the solution settles. This changes nothing in what
   * is reported: the optimality conditions are always checked on the whole
   * index space before stopping, the gradient of what had been left out
   * being recomputed for the purpose. Defaults to 1. */
  intLastAlgParSMOS  ///< 1st allowed new int parameter for derived classes
  };

 using Solver::set_par;  // keep the other set_par() overloads visible

 /// honoured parameter: dblSMOTol
 void set_par( idx_type par , double value ) override {
  if( par == dblSMOTol ) { f_tol = value; return; }
  Solver::set_par( par , value );
  }

 /// honoured parameters: intMaxIter, intSMOPath, intSMOShrink
 void set_par( idx_type par , int value ) override {
  if( par == intMaxIter ) { f_max_iter = value; return; }
  if( par == intSMOPath ) { f_path = bool( value ); return; }
  if( par == intSMOShrink ) { f_shrink = bool( value ); return; }
  Solver::set_par( par , value );
  }

 [[nodiscard]] double get_dbl_par( idx_type par ) const override {
  if( par == dblSMOTol ) return( f_tol );
  return( Solver::get_dbl_par( par ) );
  }

 [[nodiscard]] int get_int_par( idx_type par ) const override {
  if( par == intMaxIter ) return( f_max_iter );
  if( par == intSMOPath ) return( int( f_path ) );
  if( par == intSMOShrink ) return( int( f_shrink ) );
  return( Solver::get_int_par( par ) );
  }

 [[nodiscard]] idx_type get_num_int_par( void ) const override {
  return( Solver::get_num_int_par() + intLastAlgParSMOS - intLastAlgPar );
  }

 [[nodiscard]] int get_dflt_int_par( idx_type par ) const override {
  if( ( par == intSMOShrink ) || ( par == intSMOPath ) )
   return( 1 );
  return( Solver::get_dflt_int_par( par ) );
  }

 [[nodiscard]] idx_type int_par_str2idx( const std::string & name )
  const override {
  if( name == "intSMOPath" ) return( intSMOPath );
  if( name == "intSMOShrink" ) return( intSMOShrink );
  return( Solver::int_par_str2idx( name ) );
  }

 [[nodiscard]] const std::string & int_par_idx2str( idx_type idx )
  const override {
  static const std::string path = "intSMOPath" , shrink = "intSMOShrink";
  if( idx == intSMOPath ) return( path );
  if( idx == intSMOShrink ) return( shrink );
  return( Solver::int_par_idx2str( idx ) );
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

 /// unlearns a sample, leaving the solution optimal without it
 /** Drives to zero the multipliers of the sample \p i keeping every other
  * one at its own optimality condition, so that what is left is the exact
  * solution of the training problem *without* that sample, at the cost of
  * one walk along the solution path rather than of a training [see
  * follow_path()]. The sample stays in the SVMBlock and nothing of it is
  * changed: what changes is the solution this Solver holds, which becomes
  * the one of the problem the sample does not take part in. That is what
  * makes the leave-one-out estimate, and the k-fold that unlearns one fold
  * at a time, cost a walk each instead of a training each.
  *
  * The Solver has to have a solution to start from, i.e., compute() must
  * have been called and have found one. Returns kOK if the sample has been
  * unlearnt and kError if the path could not be followed, in which case the
  * multipliers are feasible but they are not the solution asked for, and a
  * compute() puts things back in order. */

 int unlearn( Index i );

/*--------------------------------------------------------------------------*/
 /// returns the number of iterations of the last call to compute()

 Index get_iter( void ) const { return( f_iter ); }

/*--------------------------------------------------------------------------*/
 /// the iterations of the last call to compute(), for whoever asks generically

 [[nodiscard]] long get_elapsed_iterations( void ) const override {
  return( long( f_iter ) );
  }

/*---------------------- PROTECTED PART OF THE CLASS -----------------------*/

 protected:

/*--------------------------- PROTECTED METHODS ----------------------------*/

 /// tells whether the given Modification allows a warm start
 /** Returns true if the given Modification, and recursively all those a
  * GroupModification contains, leave the multipliers of the previous call to
  * compute() worth starting from, and false if the algorithm rather has to
  * restart from the origin. */

 bool guts_of_poM( const Modification * mod );

/*--------------------------------------------------------------------------*/
 /// composes into v_smap the sample map of a Modification changing the data

 void compose_smap( const Modification * mod );

/*--------------------------------------------------------------------------*/
 /// realigns the multipliers to a dual index space that has changed size
 /** Realigns the multipliers, and with them everything else this Solver
  * caches, to the dual index space of a data set some samples have been
  * added to or removed from, which is what v_smap describes: the multiplier
  * of a dual index that survives is kept, that of a new one starts at zero,
  * the equality constraint is made to hold again and the gradient is
  * recomputed. Returns false if the previous multipliers turn out not to be
  * worth starting from, in which case the caller reload()s. */

 bool resample( void );

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
 /// fills the cached diagonal of the Hessian of the dual

 void fill_diagonal( void );

/*--------------------------------------------------------------------------*/
 /// makes the multipliers satisfy the equality constraint again
 /** Moves the multipliers so that s^T alpha is the right-hand side of the
  * equality constraint, updating the gradient accordingly, and returns true
  * if it succeeds: it does not if the bounds leave no room for it, which
  * means that the dual is empty, hence that the training problem is
  * unbounded. It is only called if the bias is not regularised, for otherwise
  * there is no equality constraint. */

 bool restore_equality( void );

/*--------------------------------------------------------------------------*/
 /// exchanges the two entries of the current order of the dual indices
 /** Exchanges everything that is indexed by the position of a dual index in
  * the current order: the multiplier, the gradient, the diagonal, the sign,
  * the linear coefficient, the sample and the order itself. Note that the
  * Gram matrix is *not* touched, being indexed by the sample rather than by
  * the dual index. */

 void swap_index( Index a , Index b );

/*--------------------------------------------------------------------------*/
 /// takes out of the active set what cannot be selected any more
 /** Moves past the end of the active set every dual index that, given the
  * largest value \p m and the smallest value \p M of the bias that the
  * multipliers allow, can be neither of the two of a violating pair: an
  * index at a bound whose own value of the bias is on the wrong side of the
  * interval [ \p M , \p m ] is one such. The exclusion is not permanent:
  * unshrink() puts everything back before the optimality conditions are
  * declared to hold. */

 void shrink( double m , double M );

/*--------------------------------------------------------------------------*/
 /// the same, for the dual without the equality constraint
 /** Moves past the end of the active set every dual index that is at a bound
  * satisfying its own optimality condition by more than the largest
  * violation \p viol left on the active set, hence that a step of the size
  * the others still allow cannot wake up. */

 void shrink_box( double viol );

/*--------------------------------------------------------------------------*/
 /// puts back into the active set everything that had been taken out
 /** Restores the whole index space, recomputing the gradient of what had
  * been left out, which the steps taken in the meantime have made stale:
  * G = Q alpha + q costs one row of the Gram matrix per *nonzero* multiplier
  * rather than one per index restored. */

 void unshrink( void );

/*--------------------------------------------------------------------------*/
 /// puts the dual indices back in the order the SVMBlock has them
 /** Undoes the permutation the shrinking has built, so that everything the
  * Solver holds is again indexed by the dual index of the SVMBlock, which is
  * what the caller, the realignment and the solution reading all assume. */

 void restore_order( void );

/*--------------------------------------------------------------------------*/
 /// how the free multipliers and the bias react to the multiplier of c
 /** Solves the system that says how the multipliers of the *free* dual
  * indices \p S and the bias have to move, per unit of movement of the
  * multiplier of \p c, for the free indices to keep their own optimality
  * condition and the equality constraint to keep holding:
  * \f[
  *    Q_{SS} \beta + s_S \beta_b = - Q_{Sc} \quad , \quad
  *    s_S \cdot \beta = - s_c
  * \f]
  * The system is (|S|+1) x (|S|+1), or |S| x |S| when the bias is
  * regularised and there is no equality constraint, and is solved by plain
  * Gaussian elimination with partial pivoting: |S| is the number of margin
  * support vectors, hence small, and re-solving it at each event of the path
  * costs much less than what the rank-one updates of [Cauwenberghs and
  * Poggio] would cost in code. Returns false if the system is singular, in
  * which case the path cannot be followed and the caller falls back on the
  * iteration. */

 bool solve_free_system( const Subset & S , Index c , doubleVec & beta ,
                         double & beta_b ) const;

/*--------------------------------------------------------------------------*/
 /// moves the multiplier of c to \p to along the exact solution path
 /** Moves the multiplier of the dual index \p c towards \p to keeping
  * *every other* dual index at its own optimality condition, which is the
  * incremental / decremental algorithm of
  *
  *   G. Cauwenberghs, T. Poggio "Incremental and Decremental Support Vector
  *   Machine Learning" NIPS 13, 409 - 415, 2000
  *
  * The multipliers of the free indices and the bias follow the one of \p c
  * along the direction that solve_free_system() gives, up to the first
  * *event*: the multiplier of \p c reaches \p to, its own condition starts
  * holding, a free multiplier reaches a bound, or a bounded one stops
  * satisfying its condition and becomes free. At each event the direction is
  * recomputed and the walk resumes, so what is left behind is optimal at
  * every point of the path and exact when it stops.
  *
  * Growing a multiplier from zero is how a *new* sample is learnt, driving
  * it to zero is how one is *unlearnt*, and the two are the same walk taken
  * in opposite directions, hence the same code.
  *
  * Returns kOK if the path has been followed to its end and kError if it
  * could not be, in which case nothing is claimed about the multipliers save
  * that they are feasible, and the caller has to fall back on the
  * iteration. */

 int follow_path( Index c , double to );

/*--------------------------------------------------------------------------*/
 /// the value, and the bias, of the solution the path has left behind
 /** Computes what the caller of a walk along the path has to report: the
  * value of the dual at the multipliers it has left, and, when the bias is
  * regularised and is therefore no multiplier of an equality constraint but
  * one more component of the model, the bias itself. */

 void path_solution( void );

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
             ( f_K[ std::size_t( f_di[ k ] ) * f_n + f_di[ l ] ] + f_rb )
             / f_rw;
  if( k == l )
   q += f_d;
  return( q );
  }

/*---------------------------- PROTECTED FIELDS ----------------------------*/

 SVMBlock * f_SVM = nullptr;   ///< the SVMBlock to be solved

 double f_tol = 1e-3;          ///< tolerance on the optimality conditions
 int f_max_iter = -1;          ///< maximum number of iterations, < 0 = none
 bool f_shrink = true;         ///< whether the active set is shrunk
 bool f_path = true;           ///< whether the added samples are learnt

 /* How many shrinking passes are taken before the active set is restored
  * anyway: it bounds what a collapsed active set can cost in iterations,
  * which is the other half of the trade-off the shrinking is. */

 Index f_patience = 20;

 bool f_solved = false;        ///< true if a solution is available
 double f_value = 0;           ///< the value of the dual at the solution
 double f_b = 0;               ///< the bias of the model
 Index f_iter = 0;             ///< iterations of the last call to compute()

 doubleVec v_alpha;            ///< the current multipliers
 doubleVec v_G;                ///< the gradient of the dual at them

 doubleVec v_QD;               ///< the diagonal of the Hessian of the dual

 /* The active set is a *prefix* of the current order of the dual indices:
  * shrinking an index is exchanging it with the last active one, which keeps
  * every scan of the active set sequential, as it would be without
  * shrinking at all. The order is undone before compute() returns [see
  * restore_order()], so that nothing outside this class ever sees it. */

 Index f_act = 0;              ///< the size of the active set
 IndexVec v_perm;              ///< where each entry of the order came from

 // the data of the dual, cached out of the SVMBlock- - - - - - - - - - - - -

 Index f_n = 0;                ///< the number n of samples
 Index f_N = 0;                ///< the size N of the dual index space
 double f_u = 0;               ///< the upper bound on the multipliers
 double f_rb = 0;              ///< 1 if the bias is regularised, 0 otherwise
 double f_d = 0;               ///< the diagonal term due to the squared loss

 /* The weight of the regularisation term, which divides the Hessian of the
  * dual: it is 1 unless the SVMBlock is a chunk of a consensus structure,
  * where the term is split among the chunks. */

 double f_rw = 1;              ///< the weight of the regularisation term
 double f_mu = 0;              ///< the right-hand side of the equality
 double f_dc = 0;              ///< the constant term of the dual
 const double * f_K = nullptr;      ///< the n x n Gram matrix
 const Index * f_di = nullptr;      ///< the N sample indices

 /* The signs, the linear coefficients and the sample of each dual index are
  * *copied* rather than pointed to in the SVMBlock, since the gradient is
  * updated with the difference between their new and their old value, and the
  * multipliers are realigned by matching the dual indices of the two, both of
  * which require having the previous ones around once the SVMBlock has
  * changed. */

 doubleVec v_s;                ///< the N signs
 doubleVec v_q;                ///< the N linear coefficients
 IndexVec v_di_c;              ///< the N sample indices, copied

 /* Which sample of the data set this Solver is aligned to each sample of the
  * current one was, Inf< Index >() for a sample that has been added since;
  * empty when no sample has been added or removed. It is composed as the
  * Modification are popped, so that any number of changes between two calls
  * to compute() is dealt with in one go [see compose_smap()]. */

 Subset v_smap;

 /// the dual indices that have no multiplier coming from the previous data
 /// set, i.e., those the incremental path has to learn [see compute()]
 Subset v_new;

 bool f_rmvd = false;    ///< true if samples have been removed, not only added
 bool f_optimal = false; ///< true if the last call to compute() found the optimum

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
