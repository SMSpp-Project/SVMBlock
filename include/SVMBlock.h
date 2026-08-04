/*--------------------------------------------------------------------------*/
/*---------------------------- File SVMBlock.h -----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Header file for the *abstract* class SVMBlock, which implements the Block
 * concept [see Block.h] for the training problem of a Support Vector Machine,
 * and holds all the machinery that the classification and the regression
 * variants have in common: the data set, the kernel, the two supported
 * formulations of the training problem and the recovery of the model out of
 * the solution.
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

#ifndef __SVMBlock
 #define __SVMBlock
                      /* self-identification: #endif at the end of the file */

/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include "Block.h"

#include "ColVariable.h"

#include "FRealObjective.h"

#include "FRowConstraint.h"

#include "OneVarConstraint.h"

/*--------------------------------------------------------------------------*/
/*------------------------------ NAMESPACE ---------------------------------*/
/*--------------------------------------------------------------------------*/

/// namespace for the Structured Modeling System++ (SMS++)
namespace SMSpp_di_unipi_it
{
/*--------------------------------------------------------------------------*/
/*-------------------------------- CLASSES ---------------------------------*/
/*--------------------------------------------------------------------------*/
/** @defgroup SVMBlock_CLASSES Classes in SVMBlock.h
 *  @{ */

/*--------------------------------------------------------------------------*/
/*----------------------------- CLASS SVMBlock -----------------------------*/
/*--------------------------------------------------------------------------*/
/*--------------------------- GENERAL NOTES --------------------------------*/
/*--------------------------------------------------------------------------*/
/// implementation of the Block concept for the training of a SVM
/** The SVMBlock class implements the Block concept [see Block.h] for the
 * training problem of a Support Vector Machine (SVM) over a data set of
 * \f$ n \f$ samples \f$ x_i \in \mathcal{R}^m \f$ with targets \f$ y_i \f$.
 *
 * The class is abstract: the two concrete derived classes are SVCBlock, where
 * \f$ y_i \in \{ -1 , +1 \} \f$ and the model is a maximum-margin separating
 * hyperplane, and SVRBlock, where \f$ y_i \in \mathcal{R} \f$ and the model is
 * a regression function whose errors are not penalised inside a tube of given
 * half-width \f$ \epsilon \f$. Both are governed by the same machinery, since
 * both training problems are instances of the same parametric pair of
 * formulations, as detailed below.
 *
 * <b>The primal formulation.</b> The model is the affine function
 * \f$ x \mapsto \langle w , x \rangle + b \f$, and the training problem is
 * \f[
 *   \min_{ w , b , \xi } \quad \frac{1}{2} \| w \|^2
 *     \; [ \; + \; \frac{1}{2} b^2 \; ] \; + \;
 *     C \sum_{ k = 0 }^{ N - 1 } \xi_k^p
 * \f]
 * \f[
 *   s_k ( \langle w , x_{ i(k) } \rangle + b ) + \xi_k \geq r_k
 *   \quad , \quad \xi_k \geq 0 \quad , \quad k = 0 , \dots , N - 1
 * \f]
 * where \f$ p \in \{ 1 , 2 \} \f$ selects the linear or the quadratic
 * penalisation of the slacks, \f$ C > 0 \f$ trades the margin off against the
 * training error, and the bracketed term is present only if the bias is
 * regularised (see below). The problem is a convex quadratic program with a
 * *diagonal* Hessian, hence it is directly handled by any general-purpose
 * quadratic Solver. It is available only when the model is affine in the
 * input space, i.e., for the linear kernel.
 *
 * <b>The dual formulation.</b> Dualising the constraints of the primal yields
 * the Wolfe dual
 * \f[
 *   \max_{ \alpha } \quad - q^T \alpha - \frac{1}{2} \alpha^T Q \alpha
 *   \quad , \quad 0 \leq \alpha_k \leq u \quad , \quad
 *   [ \; s^T \alpha = 0 \; ]
 * \f]
 * with \f$ q_k = - r_k \f$,
 * \f[
 *   Q_{ kl } = s_k s_l \big( \mathcal{K}( x_{ i(k) } , x_{ i(l) } )
 *              \; [ \; + \; 1 \; ] \; \big)
 *              \; + \; \delta_{ kl } \frac{ p - 1 }{ 2C }
 * \f]
 * and \f$ u = C \f$ for \f$ p = 1 \f$, \f$ u = + \infty \f$ for \f$ p = 2 \f$.
 * The dual is a concave quadratic program over a box with (at most) one linear
 * equality constraint; its Hessian is dense, since it is the Gram matrix of
 * the kernel \f$ \mathcal{K} \f$ reweighted by the signs, but it never
 * involves the features explicitly, which is what makes nonlinear kernels
 * possible: this is why it is the *only* formulation available for them, the
 * primal ones requiring an explicit finite-dimensional feature map.
 *
 * Note that it is a *maximisation*, i.e., it is the Wolfe dual as it is
 * customarily written rather than the minimisation of its opposite: strong
 * duality holding since the training problem is convex, its optimal value is
 * then the very same number as that of the primal, so that all the
 * formulations agree on the value of the Objective of the SVMBlock. This is
 * also what a Solver ignoring the abstract representation, such as SMOSolver,
 * has to report.
 *
 * <b>The bias.</b> The equality constraint \f$ s^T \alpha = 0 \f$ of the dual
 * is the stationarity condition of the primal with respect to \f$ b \f$. If
 * the bias is *regularised*, i.e., it is appended to \f$ w \f$ and therefore
 * enters the regularisation term, the equality constraint disappears and the
 * dual becomes a purely box-constrained quadratic program, at the price of the
 * rank-one term \f$ s s^T \f$ in the Hessian (the bracketed \f$ + 1 \f$
 * above). Both variants are supported, selected by set_reg_bias().
 *
 * <b>The parametric map.</b> Everything above is written in terms of a *dual
 * index space* of size \f$ N \f$, of the map \f$ k \mapsto i(k) \f$ giving the
 * sample each dual index refers to, of the signs \f$ s_k \f$ and of the linear
 * coefficients \f$ q_k = - r_k \f$. The derived classes provide these, and
 * nothing else is needed to build either formulation:
 *
 * - SVCBlock has \f$ N = n \f$, \f$ i(k) = k \f$, \f$ s_k = y_k \f$ and
 *   \f$ q_k = -1 \f$, giving the hinge (\f$ p = 1 \f$) or squared hinge
 *   (\f$ p = 2 \f$) loss;
 *
 * - SVRBlock has \f$ N = 2n \f$, \f$ i(k) = k \bmod n \f$, \f$ s_k = +1 \f$
 *   for \f$ k < n \f$ and \f$ -1 \f$ otherwise, and
 *   \f$ q_k = - s_k y_{ i(k) } + \epsilon \f$, giving the
 *   \f$ \epsilon \f$-insensitive (\f$ p = 1 \f$) or squared
 *   \f$ \epsilon \f$-insensitive (\f$ p = 2 \f$) loss.
 *
 * <b>The decomposed formulation.</b> A third formulation is provided, which is
 * the primal reformulated so that it can be attacked by a Lagrangian, or
 * equivalently a Dantzig-Wolfe, decomposition. The samples are dealt out to
 * \f$ P \f$ chunks, each chunk is given its own copy \f$ ( w_p , b_p ) \f$ of
 * the model, the regularisation term is split evenly among the copies and the
 * copies are tied together by the *consensus* constraints
 * \f[
 *   \min_{ w , b , \xi } \quad \sum_{ p = 0 }^{ P - 1 } \Big[
 *     \frac{1}{2P} \big( \| w_p \|^2 \; [ \; + \; b_p^2 \; ] \big) + C
 *     \sum_{ k \in S_p } \xi_k^p \Big]
 * \f]
 * \f[
 *   s_k ( \langle w_p , x_{ i(k) } \rangle + b_p ) + \xi_k \geq r_k
 *   \quad , \quad \xi_k \geq 0 \quad , \quad k \in S_p
 * \f]
 * \f[
 *   w_p = w_{ p + 1 } \quad , \quad b_p = b_{ p + 1 }
 *   \quad , \quad p = 0 , \dots , P - 2
 * \f]
 * The reformulation is obviously exact, and it is written so that the SVMBlock
 * has no Variable of its own, one sub-Block per chunk, each one a SVMBlock of
 * the same type holding the primal of its own chunk, and only the consensus
 * constraints, which are linear and link the sub-Block: precisely the
 * structure that a generic Lagrangian Solver expects. Relaxing the consensus
 * constraints makes each chunk an independent, and much smaller, SVM training
 * problem with a linear term added to its objective.
 *
 * Two remarks on why it is written this way. First, the regularisation term is
 * *split*, rather than being left in one designated chunk: this keeps every
 * subproblem strongly convex, hence bounded, whereas a chunk carrying only its
 * loss would have an unbounded Lagrangian subproblem for all but the exactly
 * optimal multipliers. Second, the samples are dealt out to the chunks after
 * being sorted by target, so that each chunk sees samples of both classes:
 * a chunk whose dual signs are all equal has an unbounded subproblem in its
 * bias, unless the latter is regularised. Both conditions are checked.
 *
 * <b>The abstract representation.</b> Which of the three formulations is
 * generated is *not* part of the data of the SVMBlock, which encodes the
 * training problem and not the way it is written: it is decided by the
 * Configuration passed to generate_abstract_variables(), or found in the
 * BlockConfig, and defaults to the dual; see the comments to that method.
 *
 * <b>The model.</b> Whichever formulation and Solver is used, the trained
 * model is always available in the kernel expansion form
 * \f[
 *   f( x ) = \sum_{ i = 0 }^{ n - 1 } c_i \mathcal{K}( x_i , x ) + b
 *   \quad , \quad c_i = \sum_{ k \, : \, i(k) = i } s_k \alpha_k
 * \f]
 * see get_dual_coefficients(), decision_function() and predict(); for the
 * linear kernel the weight vector is also available, see get_w(). */

class SVMBlock : public Block
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*---------------------------- PUBLIC TYPES --------------------------------*/
/** @name Public types
 *  @{ */

 using doubleVec = std::vector< double >;      ///< a vector of double
 using c_doubleVec = const doubleVec;          ///< a const vector of double

 using IndexVec = std::vector< Index >;        ///< a vector of Index
 using c_IndexVec = const IndexVec;            ///< a const vector of Index

/*--------------------------------------------------------------------------*/
 /// the supported kernel functions
 /** The kernel \f$ \mathcal{K}( x , z ) \f$ computes the inner product of the
  * images of \f$ x \f$ and \f$ z \f$ in the feature space; \f$ \gamma \f$,
  * \f$ d \f$ and \f$ r \f$ are the parameters set by set_kernel(). */

 enum kernel_type {
  kLinear = 0 ,     ///< \f$ \langle x , z \rangle \f$
  kPoly = 1 ,       ///< \f$ ( \gamma \langle x , z \rangle + r )^d \f$
  kGaussian = 2 ,   ///< \f$ e^{ - \gamma \| x - z \|_2^2 } \f$
  kLaplacian = 3 ,  ///< \f$ e^{ - \gamma \| x - z \|_1 } \f$
  kSigmoid = 4      ///< \f$ \tanh( \gamma \langle x , z \rangle + r ) \f$
  };

/*--------------------------------------------------------------------------*/
 /// the supported formulations of the training problem

 enum svm_formulation {
  kWolfeDual = 0 ,  ///< the dual in the multipliers, any kernel
  kPrimal = 1 ,     ///< the primal in the weights, linear kernel only
  kDecomposed = 2   ///< the primal split over sub-Block, linear kernel only
  };

/*--------------------------------------------------------------------------*/
 /// the conventional values of gamma that are computed out of the data
 /** Any \f$ \gamma > 0 \f$ is used as it is; the two nonpositive values below
  * rather ask for \f$ \gamma \f$ to be derived from the data set, which is
  * only possible once the latter is loaded, see get_gamma(). */

 enum gamma_type {
  dGammaScale = 0 ,   ///< \f$ \gamma = 1 / ( m \, \mathrm{Var}( X ) ) \f$
  dGammaAuto = -1     ///< \f$ \gamma = 1 / m \f$
  };

/** @} ---------------------------------------------------------------------*/
/*--------------------- CONSTRUCTOR AND DESTRUCTOR -------------------------*/
/*--------------------------------------------------------------------------*/
/** @name Constructor and destructor
 *  @{ */

 /// constructor of SVMBlock, taking a pointer to the father Block
 /** Constructor of SVMBlock. It accepts a pointer to the father Block,
  * defaulting to nullptr so that this can also be used as the void
  * constructor. */

 explicit SVMBlock( Block * father = nullptr ) : Block( father ) , AR( 0 ) {}

/*--------------------------------------------------------------------------*/
 /// destructor of SVMBlock: deletes the abstract representation

 ~SVMBlock() override { guts_of_destructor(); }

/** @} ---------------------------------------------------------------------*/
/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
/*--------------------------------------------------------------------------*/
/** @name Other initializations
 *  @{ */

 /// loads the data set from memory, copying
 /** Loads the data set of the SVM, copying the data from the parameters:
  *
  * - n is the number of samples;
  *
  * - m is the number of features of each sample;
  *
  * - X is the n x m matrix of the samples, stored row-wise, so that the
  *   features of sample i are X[ i * m ] , ... , X[ i * m + m - 1 ];
  *
  * - y is the vector of the n targets, whose admissible values depend on the
  *   concrete class.
  *
  * The hyper-parameters are left untouched, so that they can be set either
  * before or after the data set. If there is any Solver attached to this
  * SVMBlock then a NBModification (the "nuclear option") is issued. */

 void load( Index n , Index m , c_doubleVec & X , c_doubleVec & y );

/*--------------------------------------------------------------------------*/
 /// loads the data set from memory, moving
 /** Like load( Index , Index , c_doubleVec & , c_doubleVec & ), but the data
  * are moved out of \p X and \p y rather than copied. */

 void load( Index n , Index m , doubleVec && X , doubleVec && y );

/*--------------------------------------------------------------------------*/
 /// loads the SVMBlock out of an istream
 /** Loads the SVMBlock out of an istream. The only supported format (frmt ==
  * 0) is the "dense" text one: the two integers n and m, followed by n rows
  * of m + 1 numbers each, the first m being the features of the sample and
  * the last one its target. Whitespace is irrelevant. The hyper-parameters
  * are not part of the format and are left untouched. */

 void load( std::istream & input , char frmt = 0 ) override;

/*--------------------------------------------------------------------------*/
 /// extends Block::deserialize( netCDF::NcGroup )
 /** Extends Block::deserialize( netCDF::NcGroup ) to the specific format of
  * the SVMBlock. Besides the "type" attribute of any :Block, the group has
  * the following dimensions, variables and attributes; the derived classes
  * add their own, see their deserialize().
  *
  * - the dimension "NSamples", containing the number n of samples; mandatory;
  *
  * - the dimension "NFeatures", containing the number m of features;
  *   mandatory;
  *
  * - the variable "X", of type double and indexed over "NSamples" and
  *   "NFeatures", containing the samples; mandatory;
  *
  * - the variable "Y", of type double and indexed over "NSamples", containing
  *   the targets; mandatory;
  *
  * - the scalar attribute "C", of type double, containing the trade-off
  *   parameter; optional, with default 1;
  *
  * - the scalar attribute "Kernel", of type int, containing one of the values
  *   of kernel_type; optional, with default kLinear;
  *
  * - the scalar attribute "Gamma", of type double, containing the parameter
  *   of the kernel, possibly one of the values of gamma_type; optional, with
  *   default dGammaScale;
  *
  * - the scalar attribute "Degree", of type int, containing the degree of the
  *   polynomial kernel; optional, with default 3;
  *
  * - the scalar attribute "Coef0", of type double, containing the constant
  *   term of the polynomial and sigmoid kernels; optional, with default 0;
  *
  * - the scalar attribute "SquaredLoss", of type int, nonzero if the slacks
  *   are penalised quadratically; optional, with default 0;
  *
  * - the scalar attribute "RegBias", of type int, nonzero if the bias is
  *   regularised together with the weights; optional, with default 0.
  *
  * Note that which formulation the abstract representation encodes is not
  * part of the format, since it is not part of the training problem: it is a
  * Configuration matter, see generate_abstract_variables(). */

 void deserialize( const netCDF::NcGroup & group ) override;

 using Block::deserialize;  // keep the other deserialize() overloads visible

/*--------------------------------------------------------------------------*/
 /// generates the abstract Variable of the SVMBlock
 /** Generates the abstract Variable of the SVMBlock. Which formulation is
  * generated is dictated by \p stvv or, if that is nullptr, by the
  * f_static_variables_Configuration of the BlockConfig, if it is set. Either
  * of them can be
  *
  * - a SimpleConfiguration< int >, whose value \p wf is the formulation;
  *
  * - a SimpleConfiguration< std::pair< int , int > >, whose first value is
  *   the formulation \p wf and whose second one is the number \f$ P \f$ of
  *   chunks of the decomposed formulation, which the others ignore.
  *
  * With neither, \p wf defaults to kWolfeDual and \f$ P \f$ to 1. The
  * admissible values of \p wf are those of svm_formulation:
  *
  * - kWolfeDual: one static Variable, the vector "alpha" of the N multipliers
  *   of the dual;
  *
  * - kPrimal: three static Variable, the vector "w" of the m weights, the
  *   scalar "b" of the bias and the vector "xi" of the N slacks. Only
  *   admissible for the linear kernel, since no other one has an explicit
  *   finite-dimensional feature map.
  *
  * - kDecomposed: no Variable at all, and one sub-Block per chunk, each one a
  *   SVMBlock of the same type holding the primal of its own chunk with the
  *   regularisation term divided by the number of chunks. Only admissible for
  *   the linear kernel, for the same reason.
  *
  * All the Variable are continuous, and the bounds are Constraint rather than
  * being set into the Variable, see generate_abstract_constraints(). */

 void generate_abstract_variables( Configuration * stvv = nullptr ) override;

/*--------------------------------------------------------------------------*/
 /// generates the abstract Constraint of the SVMBlock
 /** Generates the abstract Constraint of the formulation that
  * generate_abstract_variables() has generated, which must therefore have
  * been called beforehand.
  *
  * For the dual formulation these are the static group "box" of N
  * LB0Constraint imposing \f$ 0 \leq \alpha_k \leq u \f$ and, unless the bias
  * is regularised, the static FRowConstraint "eq" imposing
  * \f$ s^T \alpha = 0 \f$.
  *
  * For the primal formulation these are the static group "xibox" of N
  * LB0Constraint imposing \f$ \xi_k \geq 0 \f$ and the static group "cons" of
  * N FRowConstraint imposing
  * \f$ s_k ( \langle w , x_{ i(k) } \rangle + b ) + \xi_k \geq r_k \f$.
  *
  * For the decomposed formulation these are the Constraint of each sub-Block
  * plus, in the SVMBlock proper, the static group "link" of the
  * \f$ ( P - 1 )( m + 1 ) \f$ FRowConstraint imposing the consensus, which are
  * the only ones linking the sub-Block.
  *
  * The Configuration is not used. */

 void generate_abstract_constraints( Configuration * stcc = nullptr )
  override;

/*--------------------------------------------------------------------------*/
 /// generates the abstract Objective of the SVMBlock
 /** Generates the (minimisation) Objective of the formulation that
  * generate_abstract_variables() has generated, which must therefore have
  * been called beforehand: a QuadFunction for the dual, which is *maximised*,
  * a DQuadFunction for the primal, which is minimised, and, for the
  * decomposed formulation, the Objective of each
  * sub-Block plus an *empty* one, which a Solver flattening the whole tree
  * needs to know the sense of the problem and a Lagrangian one tolerates
  * since it depends on no Variable. The Configuration is not used.
  *
  * Note that the Hessian of the dual is dense, so that generating it costs
  * \f$ O( N^2 ) \f$ time and memory; this is unavoidable for a Solver working
  * on the abstract representation, but it is *not* paid by a Solver that
  * reads the data out of the physical representation, such as SMOSolver. */

 void generate_objective( Configuration * objc = nullptr ) override;

/*--------------------------------------------------------------------------*/
 /// extends Block::serialize( netCDF::NcGroup )
 /** Extends Block::serialize( netCDF::NcGroup ) to the specific format of the
  * SVMBlock, see deserialize( netCDF::NcGroup ) for the format. */

 void serialize( netCDF::NcGroup & group ) const override;

 using Block::serialize;  // keep the other serialize() overloads visible

/*--------------------------------------------------------------------------*/
 /// returns a Solution object able to save the solution of the SVMBlock
 /** Returns a Solution object able to save the solution of the abstract
  * representation of the SVMBlock, which is what a Solver working on it
  * stores the solution of the SVMBlock into. Which one is returned is
  * dictated by the int value of \p solc, if it is a SimpleConfiguration<
  * int > (or, failing that, of the solution Configuration of the
  * BlockConfig): 1 for a RowConstraintSolution, 2 for a ColRowSolution and
  * anything else, the default, for a ColVariableSolution. */

 Solution * get_Solution( Configuration * solc = nullptr ,
                          bool emptys = true ) override;

/** @} ---------------------------------------------------------------------*/
/*----------------- METHODS FOR MODIFYING THE SVMBlock ---------------------*/
/*--------------------------------------------------------------------------*/
/** @name Modifying the hyper-parameters
 *
 * The hyper-parameters are not part of the "model" in the SMS++ sense: they
 * define which optimization problem the SVMBlock encodes. They can therefore
 * only be changed while no abstract representation is constructed and no
 * Solver is attached; otherwise, exception is thrown.
 *  @{ */

 /// sets the trade-off parameter C, which must be positive

 void set_C( double C );

/*--------------------------------------------------------------------------*/
 /// sets the kernel function and its parameters
 /** Sets the kernel: \p type is one of the values of kernel_type, \p gamma is
  * either a positive value or one of the values of gamma_type, \p degree is
  * the (positive) degree of the polynomial kernel and \p coef0 the constant
  * term of the polynomial and sigmoid ones. */

 void set_kernel( int type , double gamma = dGammaScale , int degree = 3 ,
                  double coef0 = 0 );

/*--------------------------------------------------------------------------*/
 /// sets whether the slacks are penalised quadratically

 void set_squared_loss( bool squared );

/*--------------------------------------------------------------------------*/
 /// sets whether the bias is regularised together with the weights

 void set_reg_bias( bool reg );

/*--------------------------------------------------------------------------*/
 /// sets the weight of the regularisation term
 /** Sets the weight of the regularisation term of the primal, which defaults
  * to 1. It exists so that the decomposed formulation can divide the term
  * evenly among the copies of the model it creates, and there is little
  * reason to set it by hand: doing so changes the problem that the SVMBlock
  * encodes, since the trade-off with the loss term is what \p C is for. */

 void set_reg_weight( double weight );

/** @} ---------------------------------------------------------------------*/
/*------------- METHODS FOR READING THE DATA OF THE SVMBlock ---------------*/
/*--------------------------------------------------------------------------*/
/** @name Reading the data of the SVMBlock
 *  @{ */

 /// returns the number n of samples of the data set

 Index get_NSamples( void ) const { return( f_n ); }

 /// returns the number m of features of each sample

 Index get_NFeatures( void ) const { return( f_m ); }

 /// returns the size N of the dual index space

 Index get_NDual( void ) const { return( v_ds.size() ); }

 /// returns the n x m matrix of the samples, stored row-wise

 c_doubleVec & get_X( void ) const { return( v_X ); }

 /// returns a pointer to the m features of sample i

 const double * get_x( Index i ) const { return( v_X.data() + i * f_m ); }

 /// returns the vector of the n targets

 c_doubleVec & get_y( void ) const { return( v_y ); }

 /// returns the trade-off parameter C

 double get_C( void ) const { return( f_C ); }

 /// returns the upper bound on the multipliers, C or +infinity

 double get_ub( void ) const;

 /// returns the type of the kernel, one of the values of kernel_type

 int get_kernel_type( void ) const { return( f_kernel ); }

 /// returns the value of the parameter gamma of the kernel
 /** Returns the value of the parameter \f$ \gamma \f$ of the kernel,
  * computing it out of the data set if one of the values of gamma_type has
  * been set. */

 double get_gamma( void ) const;

 /// returns the degree of the polynomial kernel

 int get_degree( void ) const { return( f_degree ); }

 /// returns the constant term of the polynomial and sigmoid kernels

 double get_coef0( void ) const { return( f_coef0 ); }

 /// returns true if the slacks are penalised quadratically

 bool get_squared_loss( void ) const { return( f_squared_loss ); }

 /// returns true if the bias is regularised together with the weights

 bool get_reg_bias( void ) const { return( f_reg_bias ); }

 /// returns the formulation the abstract representation encodes
 /** Returns which of the values of svm_formulation the abstract
  * representation encodes, or -1 if none has been generated yet. Note that
  * this need not be the value of get_formulation(), which is only the
  * default that generate_abstract_variables() uses when no Configuration
  * dictates a different one. */

 int get_generated_formulation( void ) const
 {
  if( ! ( AR & HasVar ) )
   return( -1 );
  if( AR & DecompF )
   return( kDecomposed );
  return( ( AR & PrimalF ) ? kPrimal : kWolfeDual );
  }

 /// returns the number of chunks the decomposed formulation was generated with

 Index get_num_chunk( void ) const { return( f_nchunk ); }

 /// returns the weight of the regularisation term

 double get_reg_weight( void ) const { return( f_reg_weight ); }

 /// returns the indices of the samples of chunk p of the decomposed one

 c_IndexVec & get_chunk( Index p ) const { return( v_chunk[ p ] ); }

 /// returns the half-width of the insensitivity tube, 0 if there is none

 virtual double get_epsilon( void ) const { return( 0 ); }

 /// returns the vector of the N signs s_k

 c_doubleVec & get_dual_signs( void ) const { return( v_ds ); }

 /// returns the vector of the N sample indices i( k )

 c_IndexVec & get_dual_samples( void ) const { return( v_di ); }

 /// returns the vector of the N linear coefficients q_k

 c_doubleVec & get_dual_costs( void ) const { return( v_dq ); }

/*--------------------------------------------------------------------------*/
 /// returns the kernel of two samples of the data set

 double kernel( Index i , Index j ) const
 {
  return( kernel( get_x( i ) , get_x( j ) ) );
  }

/*--------------------------------------------------------------------------*/
 /// returns the kernel of two vectors of m features

 double kernel( const double * x , const double * z ) const;

/*--------------------------------------------------------------------------*/
 /// returns the n x n Gram matrix of the kernel, stored row-wise
 /** Returns the Gram matrix \f$ \mathcal{K}( x_i , x_j ) \f$ of the data set,
  * stored row-wise. The matrix is computed the first time it is asked for and
  * cached afterwards, which costs \f$ O( n^2 m ) \f$ time and \f$ O( n^2 ) \f$
  * memory; use kernel() instead if the whole matrix is not needed. */

 c_doubleVec & get_K( void ) const;

/** @} ---------------------------------------------------------------------*/
/*------------------ METHODS FOR READING THE TRAINED MODEL -----------------*/
/*--------------------------------------------------------------------------*/
/** @name Reading the trained model
 *  @{ */

 /// sets the multipliers and the bias defining the model
 /** Sets the "physical" solution of the SVMBlock: the N multipliers \p alpha
  * and the bias \p b. This is what a Solver that does not work on the
  * abstract representation, such as SMOSolver, uses to write its solution
  * back into the SVMBlock. */

 void set_dual_solution( doubleVec && alpha , double b );

/*--------------------------------------------------------------------------*/
 /// reads the multipliers out of the abstract representation
 /** Reads the solution out of the abstract representation into the "physical"
  * one, so that the model can be evaluated: the multipliers are read out of
  * the Variable of the dual formulation, or recovered out of the slacks and
  * the weights of the primal one, and the bias is set accordingly. Throws
  * exception if no abstract representation exists. */

 void get_solution_from_abstract( void );

/*--------------------------------------------------------------------------*/
 /// writes the model into the abstract representation
 /** The inverse of get_solution_from_abstract(): writes the model currently
  * stored in the SVMBlock into the Variable of whichever formulation the
  * abstract representation encodes, i.e., into the multipliers of the dual,
  * into the weights, the bias and the slacks of the primal, or into those of
  * every sub-Block of the decomposed one. It does nothing if no abstract
  * representation exists.
  *
  * This is what a Solver that does not work on the abstract representation,
  * such as SMOSolver, uses to leave its solution where any other Solver would
  * have left it. */

 void set_solution_in_abstract( void );

/*--------------------------------------------------------------------------*/
 /// returns the N multipliers defining the model

 c_doubleVec & get_alphas( void ) const { return( v_alpha ); }

/*--------------------------------------------------------------------------*/
 /// returns the bias of the model

 double get_b( void ) const { return( f_b ); }

/*--------------------------------------------------------------------------*/
 /// returns the n coefficients of the kernel expansion of the model
 /** Returns the vector of the n coefficients
  * \f$ c_i = \sum_{ k : i(k) = i } s_k \alpha_k \f$ of the kernel expansion
  * of the model; the support vectors are the samples whose coefficient is
  * nonzero. */

 c_doubleVec & get_dual_coefficients( void ) const;

/*--------------------------------------------------------------------------*/
 /// returns the m weights of the model
 /** Returns the weight vector \f$ w = \sum_i c_i x_i \f$ of the model, which
  * only exists for the linear kernel: throws exception otherwise. */

 doubleVec get_w( void ) const;

/*--------------------------------------------------------------------------*/
 /// returns the value of the decision function of the model at x
 /** Returns \f$ \sum_i c_i \mathcal{K}( x_i , x ) + b \f$, with \p x a vector
  * of m features. */

 double decision_function( const double * x ) const;

/*--------------------------------------------------------------------------*/
 /// returns the prediction of the model at x
 /** Returns the prediction of the model at \p x, a vector of m features: the
  * value of the decision function for a regression model, its sign for a
  * classification one. */

 virtual double predict( const double * x ) const = 0;

/*--------------------------------------------------------------------------*/
 /// returns the value of the objective of the Wolfe dual at \p alpha
 /** Returns the value of the objective of the Wolfe dual, which is
  * *maximised*, at the given multipliers; at the optimal ones it is therefore
  * the optimal value of the training problem itself. */

 double dual_objective( c_doubleVec & alpha ) const;

/** @} ---------------------------------------------------------------------*/
/*------------- Methods for checking the state of the SVMBlock -------------*/
/*--------------------------------------------------------------------------*/

 /// returns true if any part of the abstract representation is there

 bool anyone_there( void ) const override
 {
  return( AR ? true : Block::anyone_there() );
  }

/*---------------------- PROTECTED PART OF THE CLASS -----------------------*/

 protected:

/*--------------------------- PROTECTED METHODS ----------------------------*/

 /// fills the parametric map defining the two formulations
 /** Fills the three vectors v_ds, v_di and v_dq describing, respectively, the
  * signs \f$ s_k \f$, the sample indices \f$ i(k) \f$ and the linear
  * coefficients \f$ q_k \f$ of the dual index space, thereby defining both
  * formulations of the training problem. It is called whenever the data set
  * or any hyper-parameter entering the map changes. */

 virtual void set_dual_data( void ) = 0;

/*--------------------------------------------------------------------------*/
 /// copies the hyper-parameters of this SVMBlock into another one
 /** Copies the hyper-parameters of this SVMBlock into \p to, which is used to
  * set up the sub-Block of the decomposed formulation; the derived classes
  * extend it with their own ones. */

 virtual void copy_hyperparameters( SVMBlock * to ) const;

/*--------------------------------------------------------------------------*/
 /// deals the samples out to the chunks of the decomposed formulation
 /** Fills v_chunk with the indices of the samples of each of the f_nchunk
  * chunks, dealing them out round-robin after having sorted them by target so
  * that each chunk sees samples of both classes. */

 void make_chunks( void );

/*--------------------------------------------------------------------------*/
 /// recomputes the bias out of the current multipliers
 /** Recomputes the bias out of the current multipliers. If the bias is
  * regularised it is \f$ \sum_k s_k \alpha_k \f$, since it is then just one
  * more component of the weight vector. Otherwise, the constraint of the
  * primal is active and its slack is known at each dual index k that is
  * strictly inside its bounds, i.e., at each "free" support vector, whence
  * \f[
  *   b = s_k ( r_k - \xi_k ) - \sum_j c_j \mathcal{K}( x_j , x_{ i(k) } )
  *   \quad , \quad
  *   \xi_k = \frac{ ( p - 1 ) \alpha_k }{ 2C }
  * \f]
  * and the average over all the free support vectors is taken. If there is no
  * free support vector, the bias is left unchanged. */

 void compute_bias( void );

/*--------------------------------------------------------------------------*/
 /// checks that the data set is complete and consistent

 void check_data( void ) const;

/*--------------------------------------------------------------------------*/
 /// throws exception if the SVMBlock cannot be reconfigured

 void check_modifiable( const std::string & method ) const;

/*--------------------------------------------------------------------------*/
 /// deletes the abstract representation and the cached quantities

 void guts_of_destructor( void );

/*--------------------------------------------------------------------------*/
 /// completes load() and deserialize(): resets the caches, issues the Mod

 void guts_of_load( void );

/*--------------------------------------------------------------------------*/
 /// prints the SVMBlock on an ostream with the given verbosity

 void print( std::ostream & output , char vlvl = 0 ) const override;

/*--------------------------------------------------------------------------*/
 /// reads the hyper-parameters out of a netCDF group
 /** Reads the hyper-parameters out of a netCDF group; the derived classes
  * extend it with their own ones. It is called by deserialize() *before* the
  * parametric map is built, since the latter may depend on them. */

 virtual void deserialize_hyperparameters( const netCDF::NcGroup & group );

/*--------------------------------------------------------------------------*/
 /// writes the hyper-parameters into a netCDF group

 virtual void serialize_hyperparameters( netCDF::NcGroup & group ) const;

/*---------------------------- PROTECTED FIELDS ----------------------------*/

 Index f_n{};                ///< the number n of samples
 Index f_m{};                ///< the number m of features

 doubleVec v_X;              ///< the n x m samples, stored row-wise
 doubleVec v_y;              ///< the n targets

 double f_C = 1;             ///< the trade-off parameter C
 int f_kernel = kLinear;     ///< the type of the kernel
 double f_gamma = dGammaScale;  ///< the parameter gamma of the kernel
 int f_degree = 3;           ///< the degree of the polynomial kernel
 double f_coef0 = 0;         ///< the constant term of the kernel

 bool f_squared_loss = false;   ///< true if the slacks are squared
 bool f_reg_bias = false;       ///< true if the bias is regularised
 double f_reg_weight = 1;       ///< the weight of the regularisation term

 Index f_nchunk = 1;            ///< the chunks the decomposed one was given

 doubleVec v_ds;             ///< the N signs s_k
 IndexVec v_di;              ///< the N sample indices i( k )
 doubleVec v_dq;             ///< the N linear coefficients q_k

 mutable doubleVec v_K;      ///< the cached n x n Gram matrix

 doubleVec v_alpha;          ///< the N multipliers of the model
 doubleVec v_w_sol;          ///< the m weights of the model, if primal
 double f_b = 0;             ///< the bias of the model
 mutable doubleVec v_dcoef;  ///< the cached n kernel expansion coefficients

 // the abstract representation - - - - - - - - - - - - - - - - - - - - - - -

 std::vector< ColVariable > v_alpha_var;  ///< the N multipliers (dual)
 std::vector< LB0Constraint > v_box;      ///< the N bounds on them (dual)
 FRowConstraint f_eq;                     ///< the equality constraint (dual)

 std::vector< ColVariable > v_w;          ///< the m weights (primal)
 ColVariable f_b_var;                     ///< the bias (primal)
 std::vector< ColVariable > v_xi;         ///< the N slacks (primal)
 std::vector< LB0Constraint > v_xi_box;   ///< the N bounds on them (primal)
 std::vector< FRowConstraint > v_cons;    ///< the N constraints (primal)

 std::vector< IndexVec > v_chunk;         ///< the chunks (decomposed)
 std::vector< FRowConstraint > v_link;    ///< the consensus ones (decomposed)

 FRealObjective f_obj;                    ///< the objective

 unsigned char AR;           ///< bit-wise coded: what abstract is there

 static constexpr unsigned char HasVar = 1;
 ///< first bit of AR == 1 if the Variable have been constructed
 static constexpr unsigned char HasObj = 2;
 ///< second bit of AR == 1 if the Objective has been constructed
 static constexpr unsigned char HasCns = 4;
 ///< third bit of AR == 1 if the Constraint have been constructed
 static constexpr unsigned char PrimalF = 8;
 ///< fourth bit of AR == 1 if the generated formulation is the primal one
 static constexpr unsigned char DecompF = 16;
 ///< fifth bit of AR == 1 if the generated formulation is the decomposed one

/*--------------------------------------------------------------------------*/

 };  // end( class( SVMBlock ) )

/** @} end( group( SVMBlock_CLASSES ) ) */

/*--------------------------------------------------------------------------*/

 }  // end( namespace SMSpp_di_unipi_it )

/*--------------------------------------------------------------------------*/

#endif  /* SVMBlock.h included */

/*--------------------------------------------------------------------------*/
/*-------------------------- End File SVMBlock.h ---------------------------*/
/*--------------------------------------------------------------------------*/
