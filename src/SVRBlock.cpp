/*--------------------------------------------------------------------------*/
/*--------------------------- File SVRBlock.cpp ----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Implementation of the SVRBlock class.
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

#include "SVRBlock.h"

/*--------------------------------------------------------------------------*/
/*------------------------- NAMESPACE AND USING ----------------------------*/
/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

/*--------------------------------------------------------------------------*/
/*----------------------------- STATIC MEMBERS -----------------------------*/
/*--------------------------------------------------------------------------*/

// register SVRBlock in the Block factory

SMSpp_insert_in_factory_cpp_1( SVRBlock );

/*--------------------------------------------------------------------------*/
/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
/*--------------------------------------------------------------------------*/

void SVRBlock::set_epsilon( double epsilon , ModParam issueMod ,
                            ModParam issueAMod )
{
 static const std::string _prfx = "SVRBlock::set_epsilon: ";

 if( epsilon < 0 )
  throw( std::invalid_argument( _prfx + "epsilon must be nonnegative" ) );

 if( epsilon == f_epsilon )
  return;

 if( ! not_dry_run( issueMod ) )
  return;

 f_epsilon = epsilon;

 // epsilon only enters the linear coefficients of the parametric map, which
 // remap() takes care of
 update_abstract( remap() , issueMod , issueAMod );

 if( issue_pmod( issueMod ) )
  Block::add_Modification( std::make_shared< SVMBlockMod >(
                            this , SVMBlockMod::eChgEpsilon ) ,
                           Observer::par2chnl( issueMod ) );

 }  // end( SVRBlock::set_epsilon )

/*--------------------------------------------------------------------------*/

void SVRBlock::copy_hyperparameters( SVMBlock * to ) const
{
 SVMBlock::copy_hyperparameters( to );

 if( auto svr = dynamic_cast< SVRBlock * >( to ) )
  svr->set_epsilon( f_epsilon );

 }  // end( SVRBlock::copy_hyperparameters )

/*--------------------------------------------------------------------------*/

void SVRBlock::deserialize_hyperparameters( const netCDF::NcGroup & group )
{
 SVMBlock::deserialize_hyperparameters( group );

 ::deserialize( group , f_epsilon , "Epsilon" , true );

 if( f_epsilon < 0 )
  throw( std::invalid_argument( "SVRBlock::deserialize: epsilon must be "
                                "nonnegative" ) );

 }  // end( SVRBlock::deserialize_hyperparameters )

/*--------------------------------------------------------------------------*/

void SVRBlock::serialize_hyperparameters( netCDF::NcGroup & group ) const
{
 SVMBlock::serialize_hyperparameters( group );

 ::serialize( group , "Epsilon" , netCDF::NcDouble() , f_epsilon );

 }  // end( SVRBlock::serialize_hyperparameters )

/*--------------------------------------------------------------------------*/
/*--------------------------- PROTECTED METHODS ----------------------------*/

/*--------------------------------------------------------------------------*/

void SVRBlock::set_dual_data( void )
{
 const Index N = 2 * f_n;

 v_ds.resize( N );
 v_di.resize( N );
 v_dq.resize( N );

 /* The two multipliers of a sample are *adjacent*, rather than the two sides
  * of the tube being one block each: a sample added to the data set then adds
  * its two dual indices at the end of the dual index space, which is where a
  * dynamic Variable is added [see Block::add_dynamic_variables()], and one
  * removed takes away two adjacent ones. */

 for( Index i = 0 ; i < f_n ; ++i ) {
  v_ds[ 2 * i ] = 1;                    // the upper side of the tube
  v_di[ 2 * i ] = i;
  v_dq[ 2 * i ] = - v_y[ i ] + f_epsilon;

  v_ds[ 2 * i + 1 ] = -1;               // the lower side of the tube
  v_di[ 2 * i + 1 ] = i;
  v_dq[ 2 * i + 1 ] = v_y[ i ] + f_epsilon;
  }

 }  // end( SVRBlock::set_dual_data )

/*--------------------------------------------------------------------------*/
/*-------------------------- End File SVRBlock.cpp -------------------------*/
/*--------------------------------------------------------------------------*/
