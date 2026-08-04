/*--------------------------------------------------------------------------*/
/*---------------------------- File SVCBlock.h -----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Header file for the *concrete* class SVCBlock, which derives from SVMBlock
 * to implement the training problem of a Support Vector Classifier.
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

#ifndef __SVCBlock
 #define __SVCBlock
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
/** @defgroup SVCBlock_CLASSES Classes in SVCBlock.h
 *  @{ */

/*--------------------------------------------------------------------------*/
/*----------------------------- CLASS SVCBlock -----------------------------*/
/*--------------------------------------------------------------------------*/
/*--------------------------- GENERAL NOTES --------------------------------*/
/*--------------------------------------------------------------------------*/
/// implementation of the Block concept for the training of a SVC
/** The SVCBlock class derives from SVMBlock [see SVMBlock.h] to implement the
 * training problem of a Support Vector Classifier, i.e., of a maximum-margin
 * hyperplane separating the samples of the two classes \f$ y_i = \pm 1 \f$.
 *
 * The primal training problem is
 * \f[
 *   \min_{ w , b } \quad \frac{1}{2} \| w \|^2
 *     \; [ \; + \; \frac{1}{2} b^2 \; ] \; + \;
 *     C \sum_{ i = 0 }^{ n - 1 }
 *     \max( 0 \, , \, 1 - y_i ( \langle w , x_i \rangle + b ) )^p
 * \f]
 * i.e., the regularisation term plus the hinge loss (\f$ p = 1 \f$) or the
 * squared hinge loss (\f$ p = 2 \f$) of the samples, and the prediction at
 * \f$ x \f$ is the sign of the decision function.
 *
 * In the notation of the base class this corresponds to \f$ N = n \f$,
 * \f$ i(k) = k \f$, \f$ s_k = y_k \f$ and \f$ q_k = -1 \f$, whence the dual
 * has \f$ Q = ( \mathcal{K} \circ y y^T ) \f$, up to the terms due to the
 * regularisation of the bias and to the squared loss; see the comments to
 * SVMBlock for the details of both formulations. */

class SVCBlock : public SVMBlock
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*--------------------- CONSTRUCTOR AND DESTRUCTOR -------------------------*/
/** @name Constructor and destructor
 *  @{ */

 /// constructor of SVCBlock, taking a pointer to the father Block

 explicit SVCBlock( Block * father = nullptr ) : SVMBlock( father ) {}

/*--------------------------------------------------------------------------*/
 /// destructor of SVCBlock

 ~SVCBlock() override = default;

/** @} ---------------------------------------------------------------------*/
/*------------------ METHODS FOR READING THE TRAINED MODEL -----------------*/
/*--------------------------------------------------------------------------*/

 /// returns the predicted class of x, i.e., the sign of the decision function

 double predict( const double * x ) const override
 {
  return( decision_function( x ) >= 0 ? 1 : -1 );
  }

/*---------------------- PROTECTED PART OF THE CLASS -----------------------*/

 protected:

/*--------------------------- PROTECTED METHODS ----------------------------*/

 /// fills the parametric map of the classification problem
 /** Fills the parametric map with \f$ N = n \f$, \f$ i(k) = k \f$,
  * \f$ s_k = y_k \f$ and \f$ q_k = -1 \f$, checking that the targets are the
  * labels of the two classes. */

 void set_dual_data( void ) override;

/*----------------------- PRIVATE PART OF THE CLASS ------------------------*/

 private:

/*---------------------------- PRIVATE METHODS -----------------------------*/

 SMSpp_insert_in_factory_h;  // insert SVCBlock in the Block factory

/*--------------------------------------------------------------------------*/

 };  // end( class( SVCBlock ) )

/** @} end( group( SVCBlock_CLASSES ) ) */

/*--------------------------------------------------------------------------*/

 }  // end( namespace SMSpp_di_unipi_it )

/*--------------------------------------------------------------------------*/

#endif  /* SVCBlock.h included */

/*--------------------------------------------------------------------------*/
/*-------------------------- End File SVCBlock.h ---------------------------*/
/*--------------------------------------------------------------------------*/
