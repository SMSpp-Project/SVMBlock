/*--------------------------------------------------------------------------*/
/*---------------------------- File SVRBlock.h -----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Header file for the *concrete* class SVRBlock, which derives from SVMBlock
 * to implement the training problem of a Support Vector Regression.
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

#ifndef __SVRBlock
 #define __SVRBlock
                      /* self-identification: #endif at the end of the file */

/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include "SVMBlock.h"

/*--------------------------------------------------------------------------*/
/*------------------------------ NAMESPACE ---------------------------------*/
/*--------------------------------------------------------------------------*/

/// namespace for the Structured Modeling System++ (SMS++)
namespace SMSpp_di_unipi_it
{
/*--------------------------------------------------------------------------*/
/*-------------------------------- CLASSES ---------------------------------*/
/*--------------------------------------------------------------------------*/
/** @defgroup SVRBlock_CLASSES Classes in SVRBlock.h
 *  @{ */

/*--------------------------------------------------------------------------*/
/*----------------------------- CLASS SVRBlock -----------------------------*/
/*--------------------------------------------------------------------------*/
/*--------------------------- GENERAL NOTES --------------------------------*/
/*--------------------------------------------------------------------------*/
/// implementation of the Block concept for the training of a SVR
/** The SVRBlock class derives from SVMBlock [see SVMBlock.h] to implement the
 * training problem of a Support Vector Regression, i.e., of a model whose
 * errors are not penalised as long as they stay inside a tube of half-width
 * \f$ \epsilon \f$ around the targets \f$ y_i \in \mathcal{R} \f$.
 *
 * The primal training problem is
 * \f[
 *   \min_{ w , b } \quad \frac{1}{2} \| w \|^2
 *     \; [ \; + \; \frac{1}{2} b^2 \; ] \; + \;
 *     C \sum_{ i = 0 }^{ n - 1 }
 *     \max( 0 \, , \,
 *           | y_i - ( \langle w , x_i \rangle + b ) | - \epsilon )^p
 * \f]
 * i.e., the regularisation term plus the \f$ \epsilon \f$-insensitive loss
 * (\f$ p = 1 \f$) or the squared \f$ \epsilon \f$-insensitive loss
 * (\f$ p = 2 \f$) of the samples, and the prediction at \f$ x \f$ is the
 * decision function itself.
 *
 * Each sample gives *two* dual indices, one for each side of the tube, and the
 * two are adjacent so that a sample added to the data set adds them at the end
 * of the dual index space: in the notation of the base class,
 * \f$ N = 2n \f$, \f$ i(k) = \lfloor k / 2 \rfloor \f$, \f$ s_k = +1 \f$
 * for \f$ k \f$ even and \f$ -1 \f$ for \f$ k \f$ odd, and
 * \f$ q_k = - s_k y_{ i(k) } + \epsilon \f$, whence the dual has, up to the
 * permutation that groups the two sides, the customary two-by-two block
 * Hessian
 * \f[
 *   Q = \left[ \begin{array}{rr}
 *          \mathcal{K} & - \mathcal{K} \\
 *        - \mathcal{K} &   \mathcal{K}
 *       \end{array} \right]
 * \f]
 * up to the terms due to the regularisation of the bias and to the squared
 * loss; see the comments to SVMBlock for the details of both formulations. */

class SVRBlock : public SVMBlock
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*--------------------- CONSTRUCTOR AND DESTRUCTOR -------------------------*/
/** @name Constructor and destructor
 *  @{ */

 /// constructor of SVRBlock, taking a pointer to the father Block

 explicit SVRBlock( Block * father = nullptr ) : SVMBlock( father ) {}

/*--------------------------------------------------------------------------*/
 /// destructor of SVRBlock

 ~SVRBlock() override = default;

/** @} ---------------------------------------------------------------------*/
/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
/*--------------------------------------------------------------------------*/

 /// extends SVMBlock::deserialize( netCDF::NcGroup )
 /** Extends SVMBlock::deserialize( netCDF::NcGroup ) with the scalar
  * variable "Epsilon", of type netCDF::NcDouble, containing the half-width
  * of the insensitivity tube; optional, with default 0.1. */

 // the format is documented here, the reading is in
 // deserialize_hyperparameters()

/*--------------------------------------------------------------------------*/
 /// sets the half-width of the insensitivity tube, which must be nonnegative
 /** Sets the half-width \f$ \epsilon \f$ of the insensitivity tube, which
  * must be nonnegative. It only enters the linear coefficients \f$ q_k \f$ of
  * the parametric map, i.e., the linear part of the Objective of the dual and
  * the sides of the constraints of the primal, hence the change is a local
  * one; see the comments to the methods modifying a SVMBlock for the meaning
  * of the two ModParam. */

 void set_epsilon( double epsilon , ModParam issueMod = eNoBlck ,
                   ModParam issueAMod = eNoBlck );

/*--------------------------------------------------------------------------*/
 /// returns the half-width of the insensitivity tube

 double get_epsilon( void ) const override { return( f_epsilon ); }

/*--------------------------------------------------------------------------*/
 /// extends SVMBlock::copy_hyperparameters() with the insensitivity tube

 void copy_hyperparameters( SVMBlock * to ) const override;

/*--------------------------------------------------------------------------*/
/*------------------ METHODS FOR READING THE TRAINED MODEL -----------------*/
/*--------------------------------------------------------------------------*/

 /// returns the predicted value at x, i.e., the decision function

 double predict( const double * x ) const override
 {
  return( decision_function( x ) );
  }

/*---------------------- PROTECTED PART OF THE CLASS -----------------------*/

 protected:

/*--------------------------- PROTECTED METHODS ----------------------------*/

 /// fills the parametric map of the regression problem

 void set_dual_data( void ) override;

/*--------------------------------------------------------------------------*/
 /// extends SVMBlock::deserialize_hyperparameters() with "Epsilon"

 void deserialize_hyperparameters( const netCDF::NcGroup & group ) override;

/*--------------------------------------------------------------------------*/
 /// extends SVMBlock::serialize_hyperparameters() with "Epsilon"

 void serialize_hyperparameters( netCDF::NcGroup & group ) const override;

/*---------------------------- PROTECTED FIELDS ----------------------------*/

 double f_epsilon = 0.1;     ///< the half-width of the insensitivity tube

/*----------------------- PRIVATE PART OF THE CLASS ------------------------*/

 private:

/*---------------------------- PRIVATE METHODS -----------------------------*/

 SMSpp_insert_in_factory_h;  // insert SVRBlock in the Block factory

/*--------------------------------------------------------------------------*/

 };  // end( class( SVRBlock ) )

/** @} end( group( SVRBlock_CLASSES ) ) */

/*--------------------------------------------------------------------------*/

 }  // end( namespace SMSpp_di_unipi_it )

/*--------------------------------------------------------------------------*/

#endif  /* SVRBlock.h included */

/*--------------------------------------------------------------------------*/
/*-------------------------- End File SVRBlock.h ---------------------------*/
/*--------------------------------------------------------------------------*/
