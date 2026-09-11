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

#include "Solution.h"

#include <cstdint>

#include <list>

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
 * - SVRBlock has \f$ N = 2n \f$, the two multipliers of a sample being
 *   adjacent: \f$ i(k) = \lfloor k / 2 \rfloor \f$, \f$ s_k = +1 \f$ for
 *   \f$ k \f$ even and \f$ -1 \f$ for \f$ k \f$ odd, and
 *   \f$ q_k = - s_k y_{ i(k) } + \epsilon \f$, giving the
 *   \f$ \epsilon \f$-insensitive (\f$ p = 1 \f$) or squared
 *   \f$ \epsilon \f$-insensitive (\f$ p = 2 \f$) loss.
 *
 * <b>Splitting the training problem.</b> A data set is one thing, and its
 * samples are not subproblems: the training problem is naturally one Block
 * with no structure at all. Yet the primal can be *rewritten* as one training
 * problem per chunk of samples, each with its own copy of the model and an
 * even share of the regularisation term, the copies being tied together by
 * linear consensus constraints; relaxing those makes each chunk an
 * independent, and much smaller, SVM. Which of the two the SVMBlock is, i.e.,
 * whether it has sub-Block at all, is therefore a *structure* it can be
 * given, and it is chosen by set_structure() [see].
 *
 * <b>The abstract representation.</b> Which of the three formulations is
 * generated is *not* part of the data of the SVMBlock, which encodes the
 * training problem and not the way it is written: it is decided by the
 * Configuration passed to generate_abstract_variables(), or found in the
 * BlockConfig, and defaults to the dual; see the comments to that method.
 *
 * <b>Changing the training problem.</b> The hyper-parameters and the targets
 * can be changed at any time, also while the abstract representation is
 * constructed and a Solver is attached to the SVMBlock: the latter updates the
 * abstract representation and issues both the *physical* Modification saying
 * what exactly has changed [see SVMBlockMod] and the *abstract* ones
 * describing how the abstract representation has changed as a consequence, so
 * that a Solver reading either representation can react to it, and in
 * particular can re-optimize starting from the previous solution rather than
 * from scratch [see SMOSolver].
 *
 * Not every change can be described that way, though. Changing the kernel, or
 * whether the bias is regularised, changes the Hessian of the dual as a whole,
 * and changing the data set changes the size of the problem; there is no
 * point in describing such a change term by term, and therefore the abstract
 * representation is rebuilt from scratch and the NBModification, the "nuclear
 * option" saying that everything has to be read anew, is issued instead. Which
 * changes are of either kind is said in the comments to each method.
 *
 * <b>The model.</b> Whichever formulation and Solver is used, the trained
 * model is always available in the kernel expansion form
 * \f[
 *   f( x ) = \sum_{ i = 0 }^{ n - 1 } c_i \mathcal{K}( x_i , x ) + b
 *   \quad , \quad c_i = \sum_{ k \, : \, i(k) = i } s_k \alpha_k
 * \f]
 * see get_dual_coefficients(), decision_function() and predict(); for the
 * linear kernel the weight vector is also available, see get_w(). */

class SVMBlockSolution;  // forward declaration of SVMBlockSolution

/*--------------------------------------------------------------------------*/

class SVMBlock : public Block
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

 friend SVMBlockSolution;  ///< make SVMBlockSolution friend

/*---------------------------- PUBLIC TYPES --------------------------------*/
/** @name Public types
 *  @{ */

 using doubleVec = std::vector< double >;      ///< a vector of double
 using c_doubleVec = const doubleVec;          ///< a const vector of double

 using doubleVec_it = doubleVec::iterator;     ///< iterator in a doubleVec
 using c_doubleVec_it = doubleVec::const_iterator;
 ///< const iterator in a doubleVec

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
 /// which problem the abstract representation encodes

 enum svm_problem {
  kWolfeDual = 0 ,  ///< the Wolfe dual in the multipliers, any kernel
  kPrimal = 1       ///< the training problem itself, linear kernel only
  };

/*--------------------------------------------------------------------------*/
 /// the decompositions the training problem can be given the structure of
 /** The training problem is a sum over the samples of a loss, plus one
  * regularisation term: it can therefore be split along the samples in two
  * dual ways, and which one the SVMBlock is given is what set_structure()
  * says. Both take an explicit feature map, hence the linear kernel. */

 enum structure_type {
  kConsensus = 0 ,  ///< one copy of the model per chunk, tied by constraints
  kBenders = 1      ///< one model in the father, one loss per chunk
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

 explicit SVMBlock( Block * father = nullptr );

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
 /** Loads the SVMBlock out of an istream in one of two formats, both text
  * ones, the hyper-parameters being part of neither and therefore left
  * untouched:
  *
  * - frmt == 0, the "dense" one: the two integers n and m, followed by n rows
  *   of m + 1 numbers each, the first m being the features of the sample and
  *   the last one its target; whitespace is irrelevant;
  *
  * - frmt == 'l', the "sparse" one of LIBSVM, which is what the data sets
  *   that are distributed for benchmarking come in: one line per sample,
  *   holding its target followed by the pairs "index:value" of its nonzero
  *   features, with the indices 1-based and in increasing order. The number
  *   of features is the largest index that appears, the samples being stored
  *   dense all the same [see get_X()]: a data set with many features and few
  *   nonzeroes per sample is therefore read, but it is not stored the way it
  *   would deserve. */

 void load( std::istream & input , char frmt = 0 ) override;

/*--------------------------------------------------------------------------*/
 /// loads the SVMBlock out of an istream in the sparse format of LIBSVM
 /** The guts of load( istream , 'l' ): reads the file once into a list of
  * (index, value) pairs, the number of features being the largest index that
  * appears and therefore known only when the input ends, and then expands it
  * into the dense storage of the SVMBlock. */

 void load_sparse( std::istream & input );

/*--------------------------------------------------------------------------*/
 /// extends Block::deserialize( netCDF::NcGroup )
 /** Extends Block::deserialize( netCDF::NcGroup ) to the specific format of
  * the SVMBlock. Besides the "type" attribute of any :Block, the group has
  * the following dimensions and variables; the derived classes add their
  * own, see their deserialize().
  *
  * - the dimension "NSamples", containing the number n of samples; mandatory;
  *
  * - the dimension "NFeatures", containing the number m of features;
  *   mandatory;
  *
  * - the variable "X", of type netCDF::NcDouble and indexed over "NSamples"
  *   and "NFeatures", containing the samples; mandatory;
  *
  * - the variable "Y", of type netCDF::NcDouble and indexed over "NSamples",
  *   containing the targets; mandatory;
  *
  * - the scalar variable "C", of type netCDF::NcDouble, containing the
  *   trade-off parameter; optional, with default 1;
  *
  * - the scalar variable "Kernel", of type netCDF::NcInt, containing one of
  *   the values of kernel_type; optional, with default kLinear;
  *
  * - the scalar variable "Gamma", of type netCDF::NcDouble, containing the
  *   parameter of the kernel, possibly one of the values of gamma_type;
  *   optional, with default dGammaScale;
  *
  * - the scalar variable "Degree", of type netCDF::NcInt, containing the
  *   degree of the polynomial kernel; optional, with default 3;
  *
  * - the scalar variable "Coef0", of type netCDF::NcDouble, containing the
  *   constant term of the polynomial and sigmoid kernels; optional, with
  *   default 0;
  *
  * - the scalar variable "SquaredLoss", of type netCDF::NcInt, nonzero if
  *   the slacks are penalised quadratically; optional, with default 0;
  *
  * - the scalar variable "RegBias", of type netCDF::NcInt, nonzero if the
  *   bias is regularised together with the weights; optional, with default 0.
  *
  * Note that which formulation the abstract representation encodes is not
  * part of the format, since it is not part of the training problem: it is a
  * Configuration matter, see generate_abstract_variables(). */

 void deserialize( const netCDF::NcGroup & group ) override;

 using Block::deserialize;  // keep the other deserialize() overloads visible

/*--------------------------------------------------------------------------*/
 /// chooses the structure of the SVMBlock, i.e., whether it has sub-Block
 /** Chooses the structure of the SVMBlock out of \p strc, which says two
  * things: *which* decomposition, a structure_type value, and in how many
  * *chunks* the samples are dealt out. It is therefore a
  * SimpleConfiguration< std::pair< int , int > >, the pair being (type, P);
  * a SimpleConfiguration< int > is taken as (kConsensus, P), which is what
  * the SVMBlock had when the consensus was the only structure it knew. If
  * \p strc is nullptr the one of the BlockConfig is used, and if that is
  * nullptr too nothing is done. P == 1, the default, is the training problem
  * as one Block, with no sub-Block at all, whatever the type.
  *
  * The training problem is a sum over the samples of a loss plus one
  * regularisation term, and both structures split it along the samples: they
  * are the two dual ways of doing so, so the same instance can be attacked
  * from either side and the optimal value is of course the same.
  *
  * With kConsensus and P > 1 the SVMBlock has P sub-SVMBlock, one per chunk,
  * each
  * holding the training problem of its own samples with its own copy
  * \f$ ( w_p , b_p ) \f$ of the model and an even share of the
  * regularisation term, and the copies are tied together by the *consensus*
  * constraints
  * \f[
  *   w_p = w_{ p + 1 } \quad , \quad b_p = b_{ p + 1 }
  *   \quad , \quad p = 0 , \dots , P - 2
  * \f]
  * which are then the only Constraint the SVMBlock has of its own, it having
  * no Variable at all: precisely the structure a generic Lagrangian Solver
  * expects, so that relaxing them makes each chunk an independent, and much
  * smaller, SVM training problem with a linear term added to its objective
  * [see set_linear_term()], which is what lets the ad hoc SMOSolver be used
  * on it; equivalently, it is the Dantzig-Wolfe decomposition over the
  * chunks.
  *
  * Two remarks on why it is assembled this way. First, the regularisation
  * term is *split*, rather than being left in one designated chunk: this
  * keeps every subproblem strongly convex, hence bounded, whereas a chunk
  * carrying only its loss would have an unbounded Lagrangian subproblem for
  * all but the exactly optimal multipliers. Second, the samples are dealt out
  * to the chunks after being sorted by target, so that each chunk sees
  * samples of both classes: a chunk whose dual signs are all equal has an
  * unbounded subproblem in its bias, unless the latter is regularised. Both
  * conditions are checked, and exception is thrown if they cannot be met.
  *
  * Only the linear kernel is supported, the primal of each chunk requiring an
  * explicit finite-dimensional feature map, and for the same reason the
  * Configuration of the Variable is not used with P > 1: the chunks are
  * necessarily primal, and the SVMBlock has no problem of its own to choose
  * the formulation of. The trained model is the same in every chunk at any
  * solution satisfying the consensus constraints, hence it is read out of any
  * of them [see get_solution_from_abstract()].
  *
  * With kBenders and P > 1 the split is the other one: the model
  * \f$ ( w , b ) \f$ and the regularisation term stay in the SVMBlock,
  * which is then the *master*, and each chunk is a sub-Block holding the
  * slacks of its own samples, their margin Constraint and its share
  * \f$ C \sum_{ k \in p } \xi_k^{ \, \cdot } \f$ of the loss. The model
  * enters the Constraint of a chunk, and nothing else does, so projecting the
  * slacks out leaves the value function
  * \f[
  *   v_p( w , b ) = \min \{ \; C \sum_{ k \in p } \xi_k^{ \, \cdot }
  *     \; : \; s_k ( \langle w , x_{ i( k ) } \rangle + b ) + \xi_k
  *     \geq r_k \; , \; \xi \geq 0 \; \}
  * \f]
  * i.e., the loss of the chunk at that model, and the training problem
  * becomes the minimisation of the regularisation term plus the sum of the
  * \f$ v_p \f$: precisely the structure a generic Benders Solver expects,
  * the model being the "complicating" Variable and each chunk an independent
  * linear program in its own slacks. The chunks are ordinary Block, not
  * SVMBlock, a loss with no model of its own being no SVM training problem.
  *
  * Note that the two structures put the very same partition of the samples
  * to two opposite uses: with kConsensus a chunk holds a whole SVM and the
  * copies of the model are what has to be reconciled, with kBenders a chunk
  * holds no model at all and the loss is what has to be approximated. This
  * is what makes the two directly comparable on the same instance.
  *
  * The structure can be changed as long as the abstract representation has
  * not been generated, the sub-Block being thrown away and built anew; after
  * that it throws exception, as the Constraint and the Objective would no
  * longer make sense. */

 void set_structure( Configuration * strc = nullptr ) override;

/*--------------------------------------------------------------------------*/
 /// returns the number of chunks the samples are dealt out to, 1 if none

 Index get_NChunks( void ) const { return( f_P ); }

/*--------------------------------------------------------------------------*/
 /// returns which decomposition the structure is, a structure_type value

 int get_structure_type( void ) const { return( f_structure ); }

/*--------------------------------------------------------------------------*/
 /// returns the samples dealt out to chunk \p p, empty if there is no chunk
 /** Returns the indices of the samples that the structure deals out to chunk
  * \p p [see set_structure()]: the partition is the same whichever
  * decomposition the structure is, which is what makes the two comparable on
  * the very same instance. */

 const Subset & get_chunk( Index p ) const {
  static const Subset empty;
  return( p < v_chunk.size() ? v_chunk[ p ] : empty );
  }

/*--------------------------------------------------------------------------*/
 /// generates the abstract Variable of the SVMBlock
 /** Generates the abstract Variable of the SVMBlock. Which problem is encoded
  * is dictated by the int value \p wf obtained as follows: if \p stvv is not
  * nullptr and it is a SimpleConfiguration< int >, then \p wf is its value;
  * otherwise, if the BlockConfig is set and its
  * f_static_variables_Configuration is a SimpleConfiguration< int >, then
  * \p wf is its value; otherwise \p wf is kWolfeDual. The admissible values
  * are those of svm_problem:
  *
  * - kWolfeDual: one static Variable, the vector "alpha" of the N multipliers
  *   of the dual;
  *
  * - kPrimal: three static Variable, the vector "w" of the m weights, the
  *   scalar "b" of the bias and the vector "xi" of the N slacks. Only
  *   admissible for the linear kernel, since no other one has an explicit
  *   finite-dimensional feature map.
  *
  * All the Variable are continuous, and the bounds are Constraint rather than
  * being set into the Variable, see generate_abstract_constraints().
  *
  * With the consensus structure [see set_structure()] the SVMBlock has no
  * Variable of its own, all of them living in the chunks: what is generated
  * here is then the primal formulation of each of them, and \p stvv is not
  * used. */

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
  * The Configuration is not used. */

 void generate_abstract_constraints( Configuration * stcc = nullptr )
  override;

/*--------------------------------------------------------------------------*/
 /// generates the abstract Objective of the SVMBlock
 /** Generates the (minimisation) Objective of the formulation that
  * generate_abstract_variables() has generated, which must therefore have
  * been called beforehand: a QuadFunction for the dual, which is *maximised*,
  * and a DQuadFunction for the primal, which is minimised. The Configuration
  * is not used.
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
  * anything else, the default, for a ColVariableSolution.
  *
  * The value 3 rather asks for a SVMBlockSolution [see SVMBlockSolution.h],
  * which saves the trained model instead of the abstract representation:
  * that is what one writes to a file, since it is what outlives the training
  * problem, but it is not what a Solver working on the abstract
  * representation, or the machinery combining solutions of sub-Block, needs
  * to see, whence it is not the default. */

 Solution * get_Solution( Configuration * solc = nullptr ,
                          bool emptys = true ) override;

/** @} ---------------------------------------------------------------------*/
/*----------------- METHODS FOR MODIFYING THE SVMBlock ---------------------*/
/*--------------------------------------------------------------------------*/
/** @name Modifying the hyper-parameters and the targets
 *
 * The hyper-parameters are not part of the "model" in the SMS++ sense: they
 * define which optimization problem the SVMBlock encodes. Together with the
 * targets, they can be changed at any time, also while the abstract
 * representation is constructed and a Solver is attached; each method says
 * which part of the abstract representation it updates and which Modification
 * it issues, the two ModParam having the usual meaning [see
 * Observer::make_par()], the first one for the physical Modification and the
 * second one for the abstract ones.
 *  @{ */

 /// sets the trade-off parameter C, which must be positive
 /** Sets the trade-off parameter \f$ C \f$ between the regularisation term
  * and the training error, which must be positive.
  *
  * In the dual formulation \f$ C \f$ is the upper bound on the multipliers,
  * unless the loss is squared, in which case it rather is the diagonal term
  * \f$ 1 / 2C \f$ of the Hessian; in the primal one it is the coefficient of
  * the slacks in the Objective. Either way the change is a local one, and it
  * is described by the corresponding Modification. */

 void set_C( double C , ModParam issueMod = eNoBlck ,
             ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// sets the kernel function and its parameters
 /** Sets the kernel: \p type is one of the values of kernel_type, \p gamma is
  * either a positive value or one of the values of gamma_type, \p degree is
  * the (positive) degree of the polynomial kernel and \p coef0 the constant
  * term of the polynomial and sigmoid ones.
  *
  * The kernel is the Hessian of the dual, hence changing it changes the
  * latter as a whole: if the abstract representation encodes the dual it is
  * therefore rebuilt and a NBModification is issued. The primal does not
  * depend on the kernel, which for it can only be the linear one: exception
  * is thrown if any other one is set while the primal is generated. */

 void set_kernel( int type , double gamma = dGammaScale , int degree = 3 ,
                  double coef0 = 0 , ModParam issueMod = eNoBlck ,
                  ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// sets whether the slacks are penalised quadratically
 /** Sets whether the slacks are penalised quadratically, which in the dual
  * means removing the upper bound on the multipliers and adding the diagonal
  * term \f$ 1 / 2C \f$ to the Hessian, and in the primal means moving the
  * coefficient \f$ C \f$ of the slacks from the linear to the quadratic part
  * of the Objective: a local change in both cases. */

 void set_squared_loss( bool squared , ModParam issueMod = eNoBlck ,
                        ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// sets whether the bias is regularised together with the weights
 /** Sets whether the bias is regularised together with the weights. In the
  * primal this only means adding the term \f$ b^2 \f$ to the Objective, but
  * in the dual it makes the equality constraint disappear and adds the
  * rank-one term \f$ s s^T \f$ to the Hessian: if the abstract representation
  * encodes the dual it is therefore rebuilt and a NBModification is issued. */

 void set_reg_bias( bool reg , ModParam issueMod = eNoBlck ,
                    ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// sets the weight of the regularisation term
 /** Sets the weight of the regularisation term of the primal, which defaults
  * to 1. It exists so that set_structure() can divide the term evenly
  * among the copies of the model it creates, and there is little reason to
  * set it by hand: doing so changes the problem that the SVMBlock encodes,
  * since the trade-off with the loss term is what \p C is for. Only the
  * primal formulation is written in terms of it, hence nothing has to be done
  * if the abstract representation encodes the dual. */

 void set_reg_weight( double weight , ModParam issueMod = eNoBlck ,
                      ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// sets the linear term of the primal
 /** Sets the linear term \f$ \lambda^T w + \mu b \f$ of the primal, which
  * defaults to zero, \p lambda being either empty, i.e., zero, or a vector
  * with one entry per feature.
  *
  * The term is what the Lagrangian relaxation of the constraints linking a
  * chunk of the consensus structure to the others leaves in the subproblem of
  * that chunk [see set_structure()], \f$ ( \lambda , \mu ) \f$ being the
  * multipliers of those constraints: it is therefore no part of the training
  * problem, whence it is not serialized. In the dual it shifts the linear
  * coefficients [see get_dual_shifts()], it moves the right-hand side of the
  * equality constraint to \f$ \mu \f$ if the bias is not regularised, and
  * it adds a constant [see get_dual_constant()]; the model it yields is
  * \f$ w = ( \sum_k \alpha_k s_k x_k - \lambda ) / \rho \f$, whence the
  * explicit feature map it takes, which only the linear kernel has. */

 void set_linear_term( c_doubleVec & lambda , double mu ,
                       ModParam issueMod = eNoBlck ,
                       ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// changes the target of one sample
 /** Changes the target of sample \p i to \p ny, which must be an admissible
  * value for the concrete class; if it is not, nothing is changed and
  * exception is thrown.
  *
  * A target only enters the parametric map, i.e., the signs \f$ s_k \f$ and
  * the linear coefficients \f$ q_k \f$ of the dual index space. If only the
  * latter change, as is the case for a regression problem, the change is a
  * local one, the coefficients being the linear part of the Objective of the
  * dual and the sides of the constraints of the primal. If the signs change,
  * as is the case for a classification problem, both formulations change all
  * over, and therefore the abstract representation is rebuilt and a
  * NBModification is issued. The Gram matrix is not affected, since it only
  * depends on the samples. */

 void chg_target( double ny , Index i , ModParam issueMod = eNoBlck ,
                  ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// changes the targets of a range of samples
 /** Changes the targets of all the samples i with rng.first <= i <
  * min( rng.second , get_NSamples() ) to the corresponding value in \p ny,
  * i.e., the target of sample i becomes *( ny + i - rng.first ); see
  * chg_target() for the details. */

 void chg_targets( c_doubleVec_it ny , Range rng = INFRange ,
                   ModParam issueMod = eNoBlck ,
                   ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// changes the targets of an arbitrary subset of samples
 /** Changes the target of sample nms[ i ] to ny[ i ] for all i; as the &&
  * tells, \p nms is "consumed" by the method. If \p ordered is true then
  * \p nms is ordered by increasing Index. See chg_target() for the details. */

 void chg_targets( c_doubleVec_it ny , Subset && nms , bool ordered = false ,
                   ModParam issueMod = eNoBlck ,
                   ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// adds \p k samples at the end of the data set
 /** Adds \p k samples at the end of the data set: \p X is their k x m matrix,
  * stored row-wise as in load(), and \p y the vector of their k targets,
  * whose admissible values are those of the concrete class.
  *
  * The samples the SVMBlock already has are *not* touched: their multipliers
  * keep their value, the new ones start at zero, and what the Gram matrix
  * already holds is kept, only the entries of the new samples being computed.
  * This is what makes a training that follows an addition cost much less than
  * a training from scratch, which is the point of the whole exercise: with
  * the model of the previous data set still feasible, and optimal for all
  * but the new samples, a Solver reading the physical representation, such as
  * SMOSolver, re-optimizes rather than restarting [see the eAddSamples
  * Modification].
  *
  * The dual index space grows accordingly, i.e., by \p k multipliers for a
  * classification problem and by 2 \p k for a regression one; the abstract
  * representation, if it exists, is rebuilt, since the size of the problem
  * has changed, and a NBModification is issued alongside the physical one. */

 void add_samples( Index k , c_doubleVec & X , c_doubleVec & y ,
                   ModParam issueMod = eNoBlck ,
                   ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// removes the samples of the given Range from the data set
 /** Removes the samples i with rng.first <= i < min( rng.second ,
  * get_NSamples() ) from the data set; the ones that follow take their
  * place, hence the indices of the samples after the Range change. See
  * remove_samples( Subset ) for the details. */

 void remove_samples( Range rng , ModParam issueMod = eNoBlck ,
                      ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// removes an arbitrary subset of samples from the data set
 /** Removes the samples whose index is in \p nms, which is "consumed" as the
  * && tells and is ordered by increasing Index if \p ordered is true; the
  * samples that survive keep their relative order, hence their indices are
  * shifted down by the number of removed samples that precede them.
  *
  * As in add_samples(), the surviving samples are not touched: their
  * multipliers keep their value and the Gram matrix is compacted rather than
  * recomputed. Note that removing a sample whose multiplier is nonzero makes
  * the previous model unfeasible for the dual, the equality constraint no
  * longer holding, which is the Solver's business to deal with [see the
  * eRmvSamples Modification].
  *
  * The whole data set cannot be removed: a SVMBlock with no sample is not a
  * training problem, and load() is what replaces a data set with another. */

 void remove_samples( Subset && nms , bool ordered = false ,
                      ModParam issueMod = eNoBlck ,
                      ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// makes \p to a SVMBlock with the same hyper-parameters as this one
 /** Copies all the hyper-parameters of this SVMBlock into \p to, so that the
  * two encode the same training problem save for the data set; the derived
  * classes extend it with their own ones. It is what whoever builds a
  * SVMBlock out of another one, e.g. over a subset of its samples, uses to
  * avoid enumerating them by hand. */

 virtual void copy_hyperparameters( SVMBlock * to ) const;

/*--------------------------------------------------------------------------*/
 /// maps an abstract change back into the physical representation
 /** Intercepts the Modification reporting a change of the abstract
  * representation made from the outside, and maps back into the physical
  * representation the only one of them that the latter can express, i.e., a
  * change of the linear term of the primal [see set_linear_term()]: this is
  * what a Lagrangian Solver writes into the Objective of a chunk of the
  * consensus structure while relaxing the constraints linking it to the
  * others, and the Solver reading the physical representation [see SMOSolver]
  * has to see it. Everything else is left alone. */

 void add_Modification( sp_Mod mod , ChnlName chnl = 0 ) override;

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

 /// returns which problem the abstract representation encodes
 /** Returns which of the values of svm_problem the abstract representation
  * encodes, or -1 if none has been generated yet. With the consensus
  * structure [see set_structure()] it is the problem the *chunks* encode,
  * i.e., the primal, the SVMBlock having no Variable of its own. */

 int get_generated_problem( void ) const
 {
  if( ! ( AR & HasVar ) )
   return( -1 );
  if( AR & Consensus )
   return( kPrimal );
  return( ( AR & PrimalF ) ? kPrimal : kWolfeDual );
  }

 /// returns the weight of the regularisation term

 double get_reg_weight( void ) const { return( f_reg_weight ); }

 /// returns lambda of the linear term of the primal, empty if it is zero

 c_doubleVec & get_linear_term( void ) const { return( v_lambda ); }

 /// returns the coefficient mu of the bias in the linear term of the primal

 double get_linear_bias( void ) const { return( f_mu ); }

 /// returns true if the linear term of the primal is not zero

 bool has_linear_term( void ) const
  { return( ( ! v_lambda.empty() ) || ( f_mu != 0 ) ); }

 /// returns the shifts the linear term applies to the linear term of the dual
 /** Fills \p shift with the get_NDual() values
  * \f[
  *   \sigma_k = s_k ( \langle \lambda , x_{ i( k ) } \rangle
  *                    [ \; + \mu \; ] ) / \rho
  * \f]
  * that the linear term of the primal subtracts from the linear coefficients
  * \f$ q_k \f$ of the dual, \f$ \rho \f$ being the weight of the
  * regularisation term and the term in \f$ \mu \f$ being there only if the
  * bias is regularised, for otherwise \f$ \mu \f$ rather is the right-hand
  * side of the equality constraint. All the shifts are zero, and \p shift is
  * only resized, if there is no linear term. */

 void get_dual_shifts( doubleVec & shift ) const;

 /// returns the constant term the linear term of the primal adds to the dual
 /** The constant \f$ - ( \| \lambda \|^2 [ \; + \mu^2 \; ] ) /
  * ( 2 \rho ) \f$ that the linear term of the primal adds to the (maximised)
  * dual, the term in \f$ \mu \f$ being there only if the bias is
  * regularised; zero if there is no linear term. */

 double get_dual_constant( void ) const;

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

/*--------------------------------------------------------------------------*/
 /// returns the \p i-th row of the Gram matrix, \f$ n \f$ entries
 /** Returns a pointer to the \p i-th row of the Gram matrix. When the whole
  * matrix is there [see get_K()] the row is read out of it; otherwise it is
  * computed, and kept in a cache of the least recently used rows whose size
  * is what set_K_memory() says. An algorithm that reads a few rows per
  * iteration, as the decomposition methods do, therefore need not pay the
  * \f$ O( n^2 ) \f$ memory of the whole matrix.
  *
  * The cache never evicts the two rows that were asked for last, hence the
  * two pointers an algorithm holds while it updates along a pair stay valid;
  * an older one does not. */

 const double * get_K_row( Index i ) const { return( K_row( i , false ) ); }

/*--------------------------------------------------------------------------*/
 /// the entries of the rows that get_K_row() has to fill in
 /** Says which columns of a row get_K_row() is going to be read at, i.e., the
  * samples an algorithm is still working on: the rows it computes carry those
  * entries alone, the others being left as they are, which is what makes a
  * cached row cost \f$ O( |A| m ) \f$ instead of \f$ O( n m ) \f$ when the
  * active set A is a fraction of the data set. Whoever needs a row in full
  * asks get_K_row_full() for it.
  *
  * Passing a different set throws away what is cached, the rows in it having
  * been filled in for the previous one, unless \p subset says that the new
  * set is contained in the one it replaces: the entries a row then has are a
  * superset of those that will be read, hence it is still good, and a
  * shrinking algorithm keeps its rows by saying so. Each row remembers which
  * of its entries have been computed, so that going back to a larger set,
  * which passing nullptr does, costs the entries that are missing and not
  * the rows. The array is not copied, hence it has to outlive the calls to
  * get_K_row(). */

 void set_K_active( const Index * samples , Index n ,
                    bool subset = false ) const;

/*--------------------------------------------------------------------------*/
 /// returns the \p i-th row of the Gram matrix, every entry of it
 /** Same as get_K_row(), except that every one of the \f$ n \f$ entries is
  * there whatever set_K_active() says, and the row is cached like any other:
  * the entries it was missing are the only cost. */

 const double * get_K_row_full( Index i ) const
 { return( K_row( i , true ) ); }

/*--------------------------------------------------------------------------*/
 /// how much memory the rows of the Gram matrix may take, in bytes
 /** Sets how much memory get_K_row() may use for its cache, and, with it,
  * whether get_K_row() materializes the whole Gram matrix: it does when the
  * matrix fits in that budget, the whole matrix being both faster to read
  * and cheaper to compute than the rows one at a time, and it does not
  * otherwise. Zero means "never materialize it, and cache one row alone",
  * which is the least memory the algorithms can run in. The default is
  * 1 GB. */

 void set_K_memory( double bytes );

/*--------------------------------------------------------------------------*/
 /// forgets the Gram matrix and the rows of it that are cached

 void drop_K( void ) const;

/*--------------------------------------------------------------------------*/
 /// forgets the cached rows, leaving the Gram matrix where it is
 /** Forgets the rows that are cached, together with everything that says
  * which they are, and leaves the Gram matrix alone: what the cache holds is
  * laid out on the number of samples of the moment, so a change of the data
  * set invalidates it, while the matrix itself is re-laid out in place by
  * whoever makes that change. */

 void drop_K_cache( void ) const;

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
 /// sets the weights and the bias defining the model
 /** Sets the "physical" solution of the SVMBlock in the form the primal has
  * it, i.e., the m weights \p w and the bias \p b, rather than as the
  * multipliers. This is what whoever obtains the model out of a primal, such
  * as a consensus SVMBlock out of any of its chunks, uses to write it
  * back into the SVMBlock; the multipliers are then unknown, and
  * get_alphas() returns them all zero. Only meaningful for the linear
  * kernel, the only one the weights exist for. */

 void set_primal_solution( doubleVec && w , double b );

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
  * or into the weights, the bias and the slacks of the primal. It does
  * nothing if no abstract representation exists.
  *
  * This is what a Solver that does not work on the abstract representation,
  * such as SMOSolver, uses to leave its solution where any other Solver would
  * have left it. */

 void set_solution_in_abstract( void );

/*--------------------------------------------------------------------------*/
 /// returns the N multipliers defining the model

 c_doubleVec & get_alphas( void ) const;

/*--------------------------------------------------------------------------*/
 /// returns the bias of the model

 double get_b( void ) const;

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
 /// what a change requires of the abstract representation

 enum ar_update {
  eARNone = 0 ,       ///< nothing has to be done
  eARBounds = 1 ,     ///< the bounds on the multipliers have to be reset
  eARSides = 2 ,      ///< the sides of the constraints have to be reset
  eARObjective = 4 ,  ///< the coefficients of the Objective have to be reset
  eARAll = 8          ///< everything has to be rebuilt
  };

/*--------------------------------------------------------------------------*/
 /// realigns the abstract representation to a change already made
 /** Realigns the abstract representation to a change that has already been
  * made in the physical representation, \p what saying which parts of it are
  * affected [see ar_update]; it does nothing if no abstract representation is
  * constructed. */

 void update_abstract( unsigned char what , ModParam issueMod ,
                       ModParam issueAMod );

/*--------------------------------------------------------------------------*/
 /// rebuilds the parametric map after the data it depends on have changed
 /** Rebuilds the parametric map, i.e., calls set_dual_data(), and returns
  * what the change requires of the abstract representation [see ar_update]:
  * nothing if the map is the same as before, everything if the signs have
  * changed, since they are all over both formulations, and only the linear
  * coefficients of the Objective of the dual, or the sides of the constraints
  * of the primal, if the coefficients alone have. */

 unsigned char remap( void );

/*--------------------------------------------------------------------------*/
 /// like remap(), but putting the previous targets back if it fails
 /** Like remap(), but if the new targets turn out not to be admissible for
  * the concrete class the previous ones, \p o_y, are put back in place before
  * the exception is let through, so that a rejected change leaves the
  * SVMBlock exactly as it was. */

 unsigned char remap_targets( doubleVec && o_y );

/*--------------------------------------------------------------------------*/
 /// resets the bounds on the multipliers of the dual formulation

 void update_abstract_bounds( ModParam issueAMod );

/*--------------------------------------------------------------------------*/
 /// resets the sides of the constraints of the primal formulation

 void update_abstract_sides( ModParam issueAMod );

/*--------------------------------------------------------------------------*/
 /// resets the coefficients of the Objective of whichever formulation
 /** Resets all the coefficients of the Objective of whichever formulation the
  * abstract representation encodes to the values dictated by the current data
  * of the SVMBlock, with the exception of the off-diagonal ones of the dual,
  * which only depend on the Gram matrix and on the signs: whatever changes
  * those rebuilds the abstract representation instead. */

 void update_abstract_objective( ModParam issueAMod );

/*--------------------------------------------------------------------------*/
 /// rebuilds the abstract representation, issuing a NBModification
 /** Destroys whatever part of the abstract representation is constructed and
  * generates it anew, of the same formulation, out of the current data of the
  * SVMBlock, then issues the NBModification saying that everything has to be
  * read anew. This is what a change that no set of "local" Modification can
  * describe does, so that whoever is attached to the SVMBlock always finds
  * an abstract representation that agrees with the physical one. */

 void rebuild_abstract( ModParam issueMod );

/*--------------------------------------------------------------------------*/
 /// deletes the abstract representation, leaving the cached data alone

 void delete_abstract( void );

/*--------------------------------------------------------------------------*/
 /// deletes the abstract representation and the cached quantities

 void guts_of_destructor( void );

/*--------------------------------------------------------------------------*/
 /// completes load() and deserialize(): resets the caches, issues the Mod

 void guts_of_load( void );

/*--------------------------------------------------------------------------*/
 /// maps one abstract Modification back into the physical representation

 void guts_of_add_Modification( const Modification * mod );

/*--------------------------------------------------------------------------*/
 /// builds, or destroys, the \p P chunks the consensus structure is made of

 void guts_of_set_structure( int type , Index P );

/*--------------------------------------------------------------------------*/
 /// deals the samples out to the \p P chunks, filling v_chunk

 void deal_out_samples( Index P );

/*--------------------------------------------------------------------------*/
 /// the dual indices of the samples dealt out to chunk \p p
 /** Returns the dual indices of the samples that v_chunk deals out to chunk
  * \p p, in increasing order: a sample contributes all of its dual indices to
  * the same chunk, since they are the two sides of the same insensitivity
  * tube and there is nothing to be gained by splitting them. */

 Subset chunk_dual( Index p ) const;

/*--------------------------------------------------------------------------*/
 /// extends the abstract representation with \p kk new dual indices
 /** Extends the abstract representation, whichever formulation it encodes,
  * with the \p kk dual indices that have just been added at the end of the
  * dual index space: the multipliers with their bounds, their term in the
  * equality constraint and their row and column of the Hessian for the dual,
  * the slacks with their bounds, their margin constraints and their term in
  * the Objective for the primal. Nothing that was already there changes, the
  * data of the samples that were already there being untouched. */

 void add_abstract_samples( Index kk , ModParam issueAMod );

/*--------------------------------------------------------------------------*/
 /// removes the given dual indices from the abstract representation
 /** Removes from the abstract representation, whichever formulation it
  * encodes, everything that belongs to the dual indices in \p dk, which are
  * ordered by increasing index and are those of the samples that have just
  * been removed, in the dual index space as it was before. */

 void rmv_abstract_samples( Subset & dk , ModParam issueAMod );

/*--------------------------------------------------------------------------*/
 /// realigns the model to a dual index space that has changed size
 /** Realigns the multipliers of the model to the dual index space of the new
  * data set: \p o_di and \p o_ds are the sample and the sign of each *old*
  * dual index, and \p o_smpl maps each new sample to the old one it was, or
  * to Inf< Index >() if it is a new one. The multiplier of a dual index that
  * survives is kept, that of a new one is zero. */

 void remap_model( const IndexVec & o_di , const doubleVec & o_ds ,
                   const Subset & o_smpl );

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

 doubleVec v_lambda;            ///< lambda of the linear term, empty = zero
 double f_mu = 0;               ///< mu of the linear term of the primal

 doubleVec v_ds;             ///< the N signs s_k
 IndexVec v_di;              ///< the N sample indices i( k )
 doubleVec v_dq;             ///< the N linear coefficients q_k

 mutable doubleVec v_K;      ///< the cached n x n Gram matrix

 /* The rows of the Gram matrix, kept for the algorithms that read a few of
  * them per iteration rather than all of it: a Solver asks for a row [see
  * get_K_row()] and gets one out of the whole matrix when that is there, out
  * of a cache of rows otherwise, the row being computed on the spot when the
  * cache does not have it. The cache is a plain LRU: v_K_cache holds
  * f_K_slots rows of f_n entries each, v_K_slot says which row sits in each
  * slot, K_row2slot maps a row to its slot and to its position in K_lru,
  * whose front is the least recently used. */

 mutable doubleVec v_K_cache;        ///< the rows the cache holds
 mutable std::vector< Index > v_K_slot;   ///< the row each slot holds
 mutable std::list< Index > K_lru;   ///< the slots, least recently used first
 mutable std::map< Index , std::pair< Index , std::list< Index >::iterator > >
                    K_row2slot;      ///< where a row is, if it is there
 /* Which entries of a cached row have been computed: one bit per entry and
  * per slot, so that a row that is asked for again, for a set that has
  * entries the previous one had not, is completed rather than recomputed.
  * It costs an eighth of a byte per entry against the eight bytes the entry
  * itself takes. */

 mutable std::vector< std::uint64_t > v_K_mask;
 mutable std::size_t f_K_words = 0;  ///< the words one row's mask takes
 mutable Index f_K_slots = 0;        ///< how many rows the cache holds
 mutable const Index * f_K_act = nullptr;  ///< the samples still worked on
 mutable Index f_K_nact = 0;         ///< how many they are

/*--------------------------------------------------------------------------*/
 /// serves a row of the Gram matrix, all of it if \p full says so

 const double * K_row( Index i , bool full ) const;

 /// how much memory the rows of the Gram matrix may take, in bytes
 double f_K_memory = 1024 * 1024 * 1024.0;
 mutable double f_gamma_res = 0;
 ///< the cached value of gamma derived from the data, 0 if not derived yet

 /* The trained model, i.e., the multipliers, the weights and the bias.
  * These are a datum of the SVMBlock rather than a snapshot of a solution:
  * they are what predict() and decision_function() evaluate, and they
  * survive the Solver that produced them. Yet they are exactly the content
  * of a SVMBlockSolution, which is therefore what holds them, so that
  * producing one is a clone() and accepting one is a copy. Never nullptr. */

 SVMBlockSolution * f_training_Results;  ///< the trained model

 mutable doubleVec v_dcoef;  ///< the cached n kernel expansion coefficients

 Index f_P = 1;              ///< the number of chunks, 1 = no sub-Block

 int f_structure = kConsensus;  ///< which decomposition the structure is

 std::vector< Subset > v_chunk;
 ///< the dual indices dealt out to each chunk, empty if there is no structure

 std::vector< FRowConstraint > v_link;  ///< the consensus constraints

 // the abstract representation - - - - - - - - - - - - - - - - - - - - - - -

 /* Whatever is indexed over the dual index space is *dynamic*, since adding
  * or removing samples changes its size: the multipliers and their bounds in
  * the dual, the slacks with their bounds and the margin constraints in the
  * primal. The weights and the bias are indexed over the features, which do
  * not change, hence they are static. */

 std::list< ColVariable > v_alpha_var;    ///< the N multipliers (dual)
 std::list< LB0Constraint > v_box;        ///< the N bounds on them (dual)
 FRowConstraint f_eq;                     ///< the equality constraint (dual)

 std::vector< ColVariable > v_w;          ///< the m weights (primal)
 ColVariable f_b_var;                     ///< the bias (primal)
 std::list< ColVariable > v_xi;           ///< the N slacks (primal)
 std::list< LB0Constraint > v_xi_box;     ///< the N bounds on them (primal)
 std::list< FRowConstraint > v_cons;      ///< the N constraints (primal)

 FRealObjective f_obj;                    ///< the objective

 unsigned char AR;           ///< bit-wise coded: what abstract is there

 static constexpr unsigned char HasVar = 1;
 ///< first bit of AR == 1 if the Variable have been constructed
 static constexpr unsigned char HasObj = 2;
 ///< second bit of AR == 1 if the Objective has been constructed
 static constexpr unsigned char HasCns = 4;
 ///< third bit of AR == 1 if the Constraint have been constructed
 static constexpr unsigned char PrimalF = 8;
 ///< fourth bit of AR == 1 if the encoded problem is the primal one
 static constexpr unsigned char Consensus = 16;

/*--------------------------------------------------------------------------*/
 /// the abstract representation is that of the Benders structure

 static constexpr unsigned char Benders = 32;
 ///< fifth bit of AR == 1 if the structure is the consensus one

/*--------------------------------------------------------------------------*/

 };  // end( class( SVMBlock ) )

/*--------------------------------------------------------------------------*/
/*-------------------------- CLASS SVMBlockMod -----------------------------*/
/*--------------------------------------------------------------------------*/
/*--------------------------- GENERAL NOTES --------------------------------*/
/*--------------------------------------------------------------------------*/
/// a change of the training problem of a SVMBlock
/** Derived class from Modification to describe a change of the data of a
 * SVMBlock, i.e., of the training problem it encodes: which one is said by
 * type(), one of the values of SVMBlock_mod_type.
 *
 * This is a *physical* Modification: it is what a Solver reading the data out
 * of the SVMBlock, such as SMOSolver, listens to, whereas one working on the
 * abstract representation rather listens to the Modification of the Variable,
 * Constraint and Objective that the SVMBlock issues alongside this one. Only
 * the changes that leave the size and the structure of the training problem
 * alone are described here; anything else is a NBModification. */

class SVMBlockMod : public Modification
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*---------------------------- PUBLIC TYPES --------------------------------*/
 /// public enum for the types of SVMBlockMod

 enum SVMBlock_mod_type {
  eChgC = 0 ,        ///< change the trade-off parameter C
  eChgKernel ,       ///< change the kernel or its parameters
  eChgSquaredLoss ,  ///< change whether the slacks are squared
  eChgRegBias ,      ///< change whether the bias is regularised
  eChgRegWeight ,    ///< change the weight of the regularisation term
  eChgLinTerm ,      ///< change the linear term of the primal
  eChgEpsilon ,      ///< change the half-width of the insensitivity tube
  eChgTargets ,      ///< change the targets of some samples
  eAddSamples ,      ///< add samples at the end of the data set
  eRmvSamples        ///< remove samples from the data set
  };

/*---------------------- CONSTRUCTOR & DESTRUCTOR --------------------------*/

 /// constructor: takes the SVMBlock and the type

 SVMBlockMod( SVMBlock * const fblock , int type )
  : f_Block( fblock ) , f_type( type ) {}

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 ~SVMBlockMod() override = default;  ///< destructor, does nothing

/*-------------------- PUBLIC METHODS OF THE CLASS -------------------------*/

 /// returns the [SVM]Block to which the SVMBlockMod refers

 [[nodiscard]] Block * get_Block( void ) const override {
  return( f_Block );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// accessor to the type of the Modification

 [[nodiscard]] int type( void ) const { return( f_type ); }

/*--------------------- PROTECTED PART OF THE CLASS ------------------------*/

 protected:

/*-------------------------- PROTECTED METHODS -----------------------------*/
 /// print the SVMBlockMod

 void print( std::ostream & output ) const override {
  output << "SVMBlockMod[" << this << "]: ";
  switch( f_type ) {
   case( eChgC ):           output << "change C"; break;
   case( eChgKernel ):      output << "change the kernel"; break;
   case( eChgSquaredLoss ): output << "change the loss"; break;
   case( eChgRegBias ):     output << "change the regularisation of the bias";
                            break;
   case( eChgLinTerm ):     output << "change the linear term"; break;
   case( eChgRegWeight ):   output << "change the regularisation weight";
                            break;
   case( eChgEpsilon ):     output << "change epsilon"; break;
   case( eChgTargets ):     output << "change the targets"; break;
   case( eAddSamples ):     output << "add samples"; break;
   case( eRmvSamples ):     output << "remove samples"; break;
   }
  }

/*--------------------- PROTECTED FIELDS OF THE CLASS ----------------------*/

 SVMBlock * f_Block;   ///< the SVMBlock the Modification refers to
 int f_type;           ///< the type of the Modification

/*--------------------------------------------------------------------------*/

 };  // end( class( SVMBlockMod ) )

/*--------------------------------------------------------------------------*/
/*------------------------ CLASS SVMBlockRngdMod ---------------------------*/
/*--------------------------------------------------------------------------*/
/// a change concerning a range of samples of a SVMBlock
/** Derived class from SVMBlockMod to describe a change concerning a range of
 * samples, i.e., all those whose index i satisfies rng().first <= i <
 * rng().second. */

class SVMBlockRngdMod : public SVMBlockMod
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*---------------------- CONSTRUCTOR & DESTRUCTOR --------------------------*/

 /// constructor: takes the SVMBlock, the type and the range

 SVMBlockRngdMod( SVMBlock * const fblock , int type , Block::Range rng )
  : SVMBlockMod( fblock , type ) , f_rng( rng ) {}

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 ~SVMBlockRngdMod() override = default;  ///< destructor, does nothing

/*-------------------- PUBLIC METHODS OF THE CLASS -------------------------*/

 /// accessor to the range

 [[nodiscard]] Block::c_Range & rng( void ) const { return( f_rng ); }

/*--------------------- PROTECTED PART OF THE CLASS ------------------------*/

 protected:

/*-------------------------- PROTECTED METHODS -----------------------------*/
 /// print the SVMBlockRngdMod

 void print( std::ostream & output ) const override {
  SVMBlockMod::print( output );
  output << " of samples [ " << f_rng.first << " , " << f_rng.second << " )"
         << std::endl;
  }

/*--------------------- PROTECTED FIELDS OF THE CLASS ----------------------*/

 Block::Range f_rng;   ///< the range of samples

/*--------------------------------------------------------------------------*/

 };  // end( class( SVMBlockRngdMod ) )

/*--------------------------------------------------------------------------*/
/*------------------------ CLASS SVMBlockSbstMod ---------------------------*/
/*--------------------------------------------------------------------------*/
/// a change concerning an arbitrary subset of samples of a SVMBlock
/** Derived class from SVMBlockMod to describe a change concerning an
 * arbitrary subset of samples, i.e., those whose index is in nms(). */

class SVMBlockSbstMod : public SVMBlockMod
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*---------------------- CONSTRUCTOR & DESTRUCTOR --------------------------*/

 /// constructor: takes the SVMBlock, the type and the subset
 /** Constructor: takes the SVMBlock, the type and the subset. As the && tells,
  * \p nms is "consumed" by the constructor and its resources become property
  * of the SVMBlockSbstMod object. */

 SVMBlockSbstMod( SVMBlock * const fblock , int type , Block::Subset && nms )
  : SVMBlockMod( fblock , type ) , f_nms( std::move( nms ) ) {}

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 ~SVMBlockSbstMod() override = default;  ///< destructor, does nothing

/*-------------------- PUBLIC METHODS OF THE CLASS -------------------------*/

 /// accessor to the subset

 [[nodiscard]] Block::c_Subset & nms( void ) const { return( f_nms ); }

/*--------------------- PROTECTED PART OF THE CLASS ------------------------*/

 protected:

/*-------------------------- PROTECTED METHODS -----------------------------*/
 /// print the SVMBlockSbstMod

 void print( std::ostream & output ) const override {
  SVMBlockMod::print( output );
  output << " of " << f_nms.size() << " samples" << std::endl;
  }

/*--------------------- PROTECTED FIELDS OF THE CLASS ----------------------*/

 Block::Subset f_nms;   ///< the subset of samples

/*--------------------------------------------------------------------------*/

 };  // end( class( SVMBlockSbstMod ) )

/*--------------------------------------------------------------------------*/
/*------------------------ CLASS SVMBlockSolution --------------------------*/
/*--------------------------------------------------------------------------*/
/*--------------------------- GENERAL NOTES --------------------------------*/
/*--------------------------------------------------------------------------*/
/// a solution of a SVMBlock, i.e., a trained model
/** The SVMBlockSolution class, derived from Solution, represents a solution
 * of a SVMBlock, i.e., the *trained model*: what the SVMBlock keeps in its
 * physical representation, that is the multipliers \f$ \alpha \f$, the
 * weights \f$ w \f$ and the bias \f$ b \f$.
 *
 * This is deliberately not the same thing as the ColVariableSolution that a
 * SVMBlock returns by default [see SVMBlock::get_Solution()], which saves the
 * Variable of whichever formulation the abstract representation encodes: that
 * one is what a Solver working on the abstract representation needs, and what
 * the machinery combining solutions of sub-Block, such as that of a
 * Lagrangian Solver, has to see. What is saved here is instead the model
 * itself, the only thing that outlives the training problem, in the form that
 * is independent of the formulation it was obtained from and that is
 * therefore the one worth writing to a file.
 *
 * Which of the two vectors is nonempty depends on where the model comes from:
 * the multipliers if it was obtained from a dual, the weights if it was
 * obtained from a primal [see SVMBlock::set_primal_solution()]. Both are
 * saved, so that the SVMBlockSolution is a faithful copy of the physical
 * representation of the SVMBlock in either case. */

class SVMBlockSolution : public Solution
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

 friend SVMBlock;  ///< make SVMBlock friend

/*------------- CONSTRUCTING AND DESTRUCTING SVMBlockSolution --------------*/
/** @name Constructing and destructing SVMBlockSolution
 *  @{ */

 /// constructor of SVMBlockSolution, it has nothing to do

 explicit SVMBlockSolution( void ) = default;

/*--------------------------------------------------------------------------*/
 // the netCDF::NcGroup methods below would otherwise hide the file-level
 // ones of the base class

 using Solution::serialize;
 using Solution::deserialize;

/*--------------------------------------------------------------------------*/
 /// deserialize a SVMBlockSolution out of a netCDF::NcGroup
 /** Deserialize a SVMBlockSolution out of a netCDF::NcGroup, which must have
  * the format described in serialize(); each of the three components is
  * optional, a missing one being taken as empty (or zero, for the bias). */

 void deserialize( const netCDF::NcGroup & group ) override final;

/*--------------------------------------------------------------------------*/
 /// destructor of SVMBlockSolution: it is virtual, and empty

 ~SVMBlockSolution() override = default;

/** @} ---------------------------------------------------------------------*/
/*------------ METHODS DESCRIBING THE BEHAVIOR OF A SVMBlockSolution -------*/
/** @name Reading and writing the model
 *  @{ */

 /// read the model of the given SVMBlock
 /** Reads the model currently stored in the given SVMBlock. If an abstract
  * representation is constructed the model is read out of it first, since
  * that is where a Solver working on it has just left it; a Solver reading
  * the physical representation, such as SMOSolver, keeps the two in synch,
  * and can anyway fill this Solution itself [see set_dual_model()]. */

 void read( const Block * block ) override final;

/*--------------------------------------------------------------------------*/
 /// write the model into the given SVMBlock
 /** Writes the model into the given SVMBlock, which must have the same data
  * set as the one it was read from, and into its abstract representation if
  * one is constructed [see SVMBlock::set_solution_in_abstract()]. */

 void write( Block * block ) override final;

/*--------------------------------------------------------------------------*/
 /// serialize a SVMBlockSolution into a netCDF::NcGroup
 /** Serialize a SVMBlockSolution into a netCDF::NcGroup, with the following
  * dimensions and variables:
  *
  * - the dimension "NMultipliers", containing the number N of multipliers,
  *   and the variable "Multipliers", of type netCDF::NcDouble and indexed
  *   over it, containing them; both are only present if the model is given
  *   by the multipliers;
  *
  * - the dimension "NFeatures", containing the number m of features, and the
  *   variable "Weights", of type netCDF::NcDouble and indexed over it,
  *   containing the weights; both are only present if the model is given by
  *   the weights;
  *
  * - the scalar variable "Bias", of type netCDF::NcDouble, containing the
  *   bias. */

 void serialize( netCDF::NcGroup & group ) const override final;

/** @} ---------------------------------------------------------------------*/
/*------------ METHODS FOR HANDLING THE "IDENTITY" OF THE Solution ---------*/
/** @name Handling the "identity" of the SVMBlockSolution
 *  @{ */

 /// returns a scaled copy of this SVMBlockSolution

 [[nodiscard]] SVMBlockSolution * scale( double factor ) const override final;

/*--------------------------------------------------------------------------*/
 /// adds a multiple of the given SVMBlockSolution to this one

 void sum( const Solution * solution , double multiplier ) override final;

/*--------------------------------------------------------------------------*/
 /// returns a copy of this SVMBlockSolution, possibly an empty one

 [[nodiscard]] SVMBlockSolution * clone( bool empty = false )
  const override final;

/** @} ---------------------------------------------------------------------*/
/*------------------------ METHODS FOR WRITING THE MODEL -------------------*/
/** @name Writing the model
 *  @{ */

 /// sets the model to the given multipliers and bias
 /** Sets the model saved in this SVMBlockSolution to the multipliers
  * \p alpha and the bias \p b, which is the form the model has when it
  * comes from a dual. This is what a Solver reading the physical
  * representation of the SVMBlock, such as SMOSolver, uses to produce a
  * Solution out of its own data, without writing anything into the SVMBlock
  * and therefore without needing any Variable to exist. */

 void set_dual_model( SVMBlock::doubleVec && alpha , double b ) {
  v_alpha = std::move( alpha );
  v_w.clear();
  f_b = b;
  }

/*--------------------------------------------------------------------------*/
 /// sets the model to the given weights and bias
 /** The counterpart of set_dual_model() for a model that comes from a primal,
  * where it is given by the weights \p w and the bias \p b and the
  * multipliers are unknown. */

 void set_primal_model( SVMBlock::doubleVec && w , double b ) {
  v_w = std::move( w );
  v_alpha.clear();
  f_b = b;
  }

/** @} ---------------------------------------------------------------------*/
/*------------------------ METHODS FOR READING THE MODEL -------------------*/
/** @name Reading the model
 *  @{ */

 /// returns the multipliers of the model, if it is given by them

 [[nodiscard]] SVMBlock::c_doubleVec & get_alphas( void ) const {
  return( v_alpha );
  }

/*--------------------------------------------------------------------------*/
 /// returns the weights of the model, if it is given by them

 [[nodiscard]] SVMBlock::c_doubleVec & get_w( void ) const {
  return( v_w );
  }

/*--------------------------------------------------------------------------*/
 /// returns the bias of the model

 [[nodiscard]] double get_b( void ) const { return( f_b ); }

/** @} ---------------------------------------------------------------------*/
/*--------------------- PROTECTED PART OF THE CLASS ------------------------*/

 protected:

/*--------------------------- PROTECTED METHODS ----------------------------*/
 /// print the SVMBlockSolution

 void print( std::ostream & output ) const override final {
  output << "SVMBlockSolution [" << this << "]: " << v_alpha.size()
         << " multipliers and " << v_w.size() << " weights" << std::endl;
  }

/*--------------------- PROTECTED FIELDS OF THE CLASS ----------------------*/

 SVMBlock::doubleVec v_alpha;   ///< the multipliers of the model
 SVMBlock::doubleVec v_w;       ///< the weights of the model
 double f_b = 0;                ///< the bias of the model

/*----------------------- PRIVATE PART OF THE CLASS ------------------------*/

 private:

/*---------------------------- PRIVATE METHODS -----------------------------*/

 SMSpp_insert_in_factory_h;  // insert SVMBlockSolution in the factory

/*--------------------------------------------------------------------------*/

 };  // end( class( SVMBlockSolution ) )

/*--------------------------------------------------------------------------*/
/*---------------- inline methods of SVMBlock needing the above ------------*/
/*--------------------------------------------------------------------------*/
// the two below are the model of the SVMBlock, which lives in the
// SVMBlockSolution it owns and is therefore only complete down here

inline SVMBlock::c_doubleVec & SVMBlock::get_alphas( void ) const {
 return( f_training_Results->get_alphas() );
 }

inline double SVMBlock::get_b( void ) const {
 return( f_training_Results->get_b() );
 }

/*--------------------------------------------------------------------------*/

/** @} end( group( SVMBlock_CLASSES ) ) */

/*--------------------------------------------------------------------------*/

 }  // end( namespace SMSpp_di_unipi_it )

/*--------------------------------------------------------------------------*/

#endif  /* SVMBlock.h included */

/*--------------------------------------------------------------------------*/
/*-------------------------- End File SVMBlock.h ---------------------------*/
/*--------------------------------------------------------------------------*/
