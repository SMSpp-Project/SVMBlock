/*--------------------------------------------------------------------------*/
/*--------------------------- File SVCBlock.cpp ----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Implementation of the SVCBlock class.
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

#include "SVCBlock.h"

#include <set>
#include <string>

/*--------------------------------------------------------------------------*/
/*------------------------- NAMESPACE AND USING ----------------------------*/
/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

/*--------------------------------------------------------------------------*/
/*----------------------------- STATIC MEMBERS -----------------------------*/
/*--------------------------------------------------------------------------*/

// register SVCBlock in the Block factory

SMSpp_insert_in_factory_cpp_1( SVCBlock );

/*--------------------------------------------------------------------------*/
/*--------------------------- PROTECTED METHODS ----------------------------*/
/*--------------------------------------------------------------------------*/

void SVCBlock::set_dual_data( void )
{
 v_ds.resize( f_n );
 v_di.resize( f_n );
 v_dq.assign( f_n , -1 );

 for( Index i = 0 ; i < f_n ; ++i ) {
  if( ( v_y[ i ] != 1 ) && ( v_y[ i ] != -1 ) )
   throw( std::invalid_argument( "SVCBlock::set_dual_data: the targets must "
                                 "be either +1 or -1" ) );
  v_ds[ i ] = v_y[ i ];
  v_di[ i ] = i;
  }

 }  // end( SVCBlock::set_dual_data )

/*--------------------------------------------------------------------------*/

void SVCBlock::labels_to_targets( doubleVec & y ) const
{
 std::set< double > labels( y.begin() , y.end() );
 if( labels.size() > 2 )
  throw( std::invalid_argument( "SVCBlock::labels_to_targets: " +
                                std::to_string( labels.size() ) + " labels, "
                                "while a binary classification has two" ) );

 // a single class is left as it is, whatever it means
 if( labels.size() < 2 )
  return;

 const double lo = *labels.begin();
 for( auto & t : y )
  t = ( t == lo ) ? -1 : 1;

 }  // end( SVCBlock::labels_to_targets )

/*--------------------------------------------------------------------------*/
/*-------------------------- End File SVCBlock.cpp -------------------------*/
/*--------------------------------------------------------------------------*/
