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

void SVRBlock::set_epsilon( double epsilon )
{
 static const std::string _prfx = "SVRBlock::set_epsilon: ";

 check_modifiable( _prfx );

 if( epsilon < 0 )
  throw( std::invalid_argument( _prfx + "epsilon must be nonnegative" ) );

 if( epsilon == f_epsilon )
  return;

 f_epsilon = epsilon;

 if( ! v_ds.empty() )  // the linear coefficients of the dual depend on it
  set_dual_data();

 }  // end( SVRBlock::set_epsilon )

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

void SVRBlock::copy_hyperparameters( SVMBlock * to ) const
{
 SVMBlock::copy_hyperparameters( to );

 if( auto svr = dynamic_cast< SVRBlock * >( to ) )
  svr->set_epsilon( f_epsilon );

 }  // end( SVRBlock::copy_hyperparameters )

/*--------------------------------------------------------------------------*/

void SVRBlock::set_dual_data( void )
{
 const Index N = 2 * f_n;

 v_ds.resize( N );
 v_di.resize( N );
 v_dq.resize( N );

 for( Index i = 0 ; i < f_n ; ++i ) {
  v_ds[ i ] = 1;                        // the upper side of the tube
  v_di[ i ] = i;
  v_dq[ i ] = - v_y[ i ] + f_epsilon;

  v_ds[ f_n + i ] = -1;                 // the lower side of the tube
  v_di[ f_n + i ] = i;
  v_dq[ f_n + i ] = v_y[ i ] + f_epsilon;
  }

 }  // end( SVRBlock::set_dual_data )

/*--------------------------------------------------------------------------*/
/*-------------------------- End File SVRBlock.cpp -------------------------*/
/*--------------------------------------------------------------------------*/
