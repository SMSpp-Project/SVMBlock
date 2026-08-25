/*--------------------------------------------------------------------------*/
/*--------------------------- File SVMBlock.cpp ----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Implementation of the SVMBlock class.
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

#include "SVMBlock.h"

#include "AbstractBlock.h"

#include "ColRowSolution.h"

#include "ColVariableSolution.h"

#include "DQuadFunction.h"

#include "LinearFunction.h"

#include "QuadFunction.h"

#include "RowConstraintSolution.h"

#include <ff/parallel_for.hpp>

#include <algorithm>

#include <cmath>

#include <iomanip>

#include <numeric>

#include <thread>

/*--------------------------------------------------------------------------*/
/*------------------------- NAMESPACE AND USING ----------------------------*/
/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

using v_coeff_pair = LinearFunction::v_coeff_pair;
using v_coeff_triple = DQuadFunction::v_coeff_triple;
using v_off_diag_term = QuadFunction::v_off_diag_term;

/*--------------------------------------------------------------------------*/
/*----------------------------- STATIC MEMBERS -----------------------------*/
/*--------------------------------------------------------------------------*/

// register SVMBlockSolution in the Solution factory; SVMBlock itself is
// abstract, hence it is the derived classes that are registered

SMSpp_insert_in_factory_cpp_0( SVMBlockSolution );

/*--------------------------------------------------------------------------*/
/*----------------------------- CONSTANTS ----------------------------------*/
/*--------------------------------------------------------------------------*/

/// the relative tolerance within which a multiplier is at one of its bounds
static constexpr double dBndEps = 1e-8;

/*--------------------------------------------------------------------------*/
/// the size of the data set above which the Gram matrix is computed in
/// parallel, below which the threads would cost more than they save

static constexpr SVMBlock::Index dParallelK = 256;

/*--------------------------------------------------------------------------*/
/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
/*--------------------------------------------------------------------------*/

void SVMBlock::load( Index n , Index m , c_doubleVec & X , c_doubleVec & y )
{
 f_n = n;
 f_m = m;
 v_X = X;
 v_y = y;

 guts_of_load();

 }  // end( SVMBlock::load( memory , copying ) )

/*--------------------------------------------------------------------------*/

void SVMBlock::load( Index n , Index m , doubleVec && X , doubleVec && y )
{
 f_n = n;
 f_m = m;
 v_X = std::move( X );
 v_y = std::move( y );

 guts_of_load();

 }  // end( SVMBlock::load( memory , moving ) )

/*--------------------------------------------------------------------------*/

void SVMBlock::load( std::istream & input , char frmt )
{
 static const std::string _prfx = "SVMBlock::load: ";

 if( frmt )
  throw( std::invalid_argument( _prfx + "unsupported format" ) );

 Index n , m;
 if( ! ( input >> eatcomments >> n ) )
  throw( std::invalid_argument( _prfx + "error reading the number of samples"
                                ) );
 if( ! ( input >> eatcomments >> m ) )
  throw( std::invalid_argument( _prfx + "error reading the number of features"
                                ) );

 doubleVec X( std::size_t( n ) * m );
 doubleVec y( n );

 for( Index i = 0 ; i < n ; ++i ) {
  for( Index j = 0 ; j < m ; ++j )
   if( ! ( input >> eatcomments >> X[ std::size_t( i ) * m + j ] ) )
    throw( std::invalid_argument( _prfx + "error reading the samples" ) );

  if( ! ( input >> eatcomments >> y[ i ] ) )
   throw( std::invalid_argument( _prfx + "error reading the targets" ) );
  }

 load( n , m , std::move( X ) , std::move( y ) );

 }  // end( SVMBlock::load( istream ) )

/*--------------------------------------------------------------------------*/

void SVMBlock::deserialize( const netCDF::NcGroup & group )
{
 static const std::string _prfx = "SVMBlock::deserialize: ";

 if( ! ::deserialize_dim( group , "NSamples" , f_n ) )
  throw( std::invalid_argument( _prfx + "NSamples dimension is required" ) );

 if( ! ::deserialize_dim( group , "NFeatures" , f_m ) )
  throw( std::invalid_argument( _prfx + "NFeatures dimension is required" ) );

 auto X = group.getVar( "X" );
 if( X.isNull() )
  throw( std::invalid_argument( _prfx + "X is required" ) );

 auto Xs = ::get_sizes_dimensions( X );
 if( ( Xs.size() != 2 ) || ( Xs[ 0 ] != f_n ) || ( Xs[ 1 ] != f_m ) )
  throw( std::invalid_argument( _prfx + "X has wrong size" ) );

 v_X.resize( std::size_t( f_n ) * f_m );
 X.getVar( v_X.data() );

 auto Y = group.getVar( "Y" );
 if( Y.isNull() )
  throw( std::invalid_argument( _prfx + "Y is required" ) );

 auto Ys = ::get_sizes_dimensions( Y );
 if( ( Ys.size() != 1 ) || ( Ys[ 0 ] != f_n ) )
  throw( std::invalid_argument( _prfx + "Y has wrong size" ) );

 v_y.resize( f_n );
 Y.getVar( v_y.data() );

 deserialize_hyperparameters( group );

 guts_of_load();

 Block::deserialize( group );

 }  // end( SVMBlock::deserialize )

/*--------------------------------------------------------------------------*/

void SVMBlock::deserialize_hyperparameters( const netCDF::NcGroup & group )
{
 static const std::string _prfx = "SVMBlock::deserialize: ";

 ::deserialize( group , f_C , "C" , true );
 if( f_C <= 0 )
  throw( std::invalid_argument( _prfx + "C must be positive" ) );

 ::deserialize( group , f_kernel , "Kernel" , true );
 if( ( f_kernel < kLinear ) || ( f_kernel > kSigmoid ) )
  throw( std::invalid_argument( _prfx + "unknown kernel type" ) );

 ::deserialize( group , f_gamma , "Gamma" , true );

 ::deserialize( group , f_degree , "Degree" , true );
 if( f_degree <= 0 )
  throw( std::invalid_argument( _prfx + "degree must be positive" ) );

 ::deserialize( group , f_coef0 , "Coef0" , true );

 int tmp = 0;
 if( ::deserialize( group , tmp , "SquaredLoss" , true ) )
  f_squared_loss = ( tmp != 0 );

 tmp = 0;
 if( ::deserialize( group , tmp , "RegBias" , true ) )
  f_reg_bias = ( tmp != 0 );

 }  // end( SVMBlock::deserialize_hyperparameters )

/*--------------------------------------------------------------------------*/

void SVMBlock::serialize( netCDF::NcGroup & group ) const
{
 Block::serialize( group );

 auto ns = group.addDim( "NSamples" , f_n );
 auto nf = group.addDim( "NFeatures" , f_m );

 ( group.addVar( "X" , netCDF::NcDouble() , { ns , nf } )
   ).putVar( v_X.data() );

 ( group.addVar( "Y" , netCDF::NcDouble() , ns ).putVar( v_y.data() ) );

 serialize_hyperparameters( group );

 }  // end( SVMBlock::serialize )

/*--------------------------------------------------------------------------*/

void SVMBlock::serialize_hyperparameters( netCDF::NcGroup & group ) const
{
 ::serialize( group , "C" , netCDF::NcDouble() , f_C );
 ::serialize( group , "Kernel" , netCDF::NcInt() , f_kernel );
 ::serialize( group , "Gamma" , netCDF::NcDouble() , f_gamma );

 if( f_kernel == kPoly )
  ::serialize( group , "Degree" , netCDF::NcInt() , f_degree );

 if( ( f_kernel == kPoly ) || ( f_kernel == kSigmoid ) )
  ::serialize( group , "Coef0" , netCDF::NcDouble() , f_coef0 );

 if( f_squared_loss )
  ::serialize( group , "SquaredLoss" , netCDF::NcInt() , int( 1 ) );

 if( f_reg_bias )
  ::serialize( group , "RegBias" , netCDF::NcInt() , int( 1 ) );

 }  // end( SVMBlock::serialize_hyperparameters )

/*--------------------------------------------------------------------------*/

void SVMBlock::guts_of_load( void )
{
 check_data();

 v_K.clear();       // the data set changed, and so did everything that is
 v_dcoef.clear();   // derived from it
 f_gamma_res = 0;

 set_dual_data();   // rebuild the parametric map

 v_alpha.assign( get_NDual() , 0 );
 v_w_sol.clear();
 f_b = 0;

 /* Whatever the abstract representation encoded is void, the size of the
  * problem having changed as well: it is generated anew out of the new data
  * set, so that whoever is attached to the SVMBlock always finds it in
  * agreement with the physical representation, and the NBModification says
  * that everything has to be read anew. */
 rebuild_abstract( eNoBlck );

 }  // end( SVMBlock::guts_of_load )

/*--------------------------------------------------------------------------*/

void SVMBlock::check_data( void ) const
{
 static const std::string _prfx = "SVMBlock::check_data: ";

 if( ( f_n == 0 ) || ( f_m == 0 ) )
  throw( std::invalid_argument( _prfx + "empty data set" ) );

 if( v_X.size() != std::size_t( f_n ) * f_m )
  throw( std::invalid_argument( _prfx + "X has wrong size" ) );

 if( v_y.size() != f_n )
  throw( std::invalid_argument( _prfx + "Y has wrong size" ) );

 }  // end( SVMBlock::check_data )

/*--------------------------------------------------------------------------*/

Solution * SVMBlock::get_Solution( Configuration * solc , bool emptys )
{
 auto config = dynamic_cast< SimpleConfiguration< int > * >( solc );

 if( ( ! config ) && f_BlockConfig )
  config = dynamic_cast< SimpleConfiguration< int > * >(
                                 f_BlockConfig->f_solution_Configuration );

 Solution * sol;
 switch( config ? config->f_value : 0 ) {
  case( 1 ): sol = new RowConstraintSolution; break;
  case( 2 ): sol = new ColRowSolution; break;
  case( 3 ): sol = new SVMBlockSolution; break;
  default:   sol = new ColVariableSolution;
  }

 if( ! emptys )
  sol->read( this );

 return( sol );

 }  // end( SVMBlock::get_Solution )

/*--------------------------------------------------------------------------*/
/*----------------- METHODS FOR MODIFYING THE SVMBlock ---------------------*/
/*--------------------------------------------------------------------------*/

void SVMBlock::set_C( double C , ModParam issueMod , ModParam issueAMod )
{
 static const std::string _prfx = "SVMBlock::set_C: ";

 if( C <= 0 )
  throw( std::invalid_argument( _prfx + "C must be positive" ) );

 if( C == f_C )
  return;

 if( ! not_dry_run( issueMod ) )
  return;

 f_C = C;

 /* C is the upper bound on the multipliers, unless the loss is squared, in
  * which case it is the diagonal of the Hessian of the dual; either way it is
  * in the Objective of the primal. The parametric map is not written in terms
  * of it in either of the concrete classes, but nothing forbids it to be. */
 unsigned char what = remap() | eARObjective;
 if( ! ( AR & PrimalF ) )
  what |= eARBounds;

 update_abstract( what , issueMod , issueAMod );

 if( issue_pmod( issueMod ) )
  Block::add_Modification( std::make_shared< SVMBlockMod >(
                            this , SVMBlockMod::eChgC ) ,
                           Observer::par2chnl( issueMod ) );

 }  // end( SVMBlock::set_C )

/*--------------------------------------------------------------------------*/

void SVMBlock::set_kernel( int type , double gamma , int degree ,
                           double coef0 , ModParam issueMod ,
                           ModParam issueAMod )
{
 static const std::string _prfx = "SVMBlock::set_kernel: ";

 if( ( type < kLinear ) || ( type > kSigmoid ) )
  throw( std::invalid_argument( _prfx + "unknown kernel type" ) );

 if( degree <= 0 )
  throw( std::invalid_argument( _prfx + "degree must be positive" ) );

 if( ( type != kLinear ) && ( AR & PrimalF ) )
  throw( std::invalid_argument( _prfx + "the primal is only available for "
                                "the linear kernel" ) );

 if( ( type == f_kernel ) && ( gamma == f_gamma ) && ( degree == f_degree )
     && ( coef0 == f_coef0 ) )
  return;

 if( ! not_dry_run( issueMod ) )
  return;

 f_kernel = type;
 f_gamma = gamma;
 f_degree = degree;
 f_coef0 = coef0;

 v_K.clear();       // the Gram matrix, if any, is no longer the right one
 f_gamma_res = 0;   // and neither is the gamma derived from the data
 v_dcoef.clear();

 /* The Gram matrix is the Hessian of the dual, hence the latter changes as a
  * whole; the primal, which only exists for the linear kernel, rather does
  * not depend on it at all. */
 update_abstract( ( AR & PrimalF ) ? eARNone : eARAll , issueMod ,
                  issueAMod );

 if( issue_pmod( issueMod ) )
  Block::add_Modification( std::make_shared< SVMBlockMod >(
                            this , SVMBlockMod::eChgKernel ) ,
                           Observer::par2chnl( issueMod ) );

 }  // end( SVMBlock::set_kernel )

/*--------------------------------------------------------------------------*/

void SVMBlock::set_squared_loss( bool squared , ModParam issueMod ,
                                 ModParam issueAMod )
{
 if( squared == f_squared_loss )
  return;

 if( ! not_dry_run( issueMod ) )
  return;

 f_squared_loss = squared;

 // the upper bound on the multipliers and the diagonal of the Hessian of the
 // dual, the linear and the quadratic coefficients of the slacks in the
 // primal
 unsigned char what = remap() | eARObjective;
 if( ! ( AR & PrimalF ) )
  what |= eARBounds;

 update_abstract( what , issueMod , issueAMod );

 if( issue_pmod( issueMod ) )
  Block::add_Modification( std::make_shared< SVMBlockMod >(
                            this , SVMBlockMod::eChgSquaredLoss ) ,
                           Observer::par2chnl( issueMod ) );

 }  // end( SVMBlock::set_squared_loss )

/*--------------------------------------------------------------------------*/

void SVMBlock::set_reg_bias( bool reg , ModParam issueMod ,
                             ModParam issueAMod )
{
 if( reg == f_reg_bias )
  return;

 if( ! not_dry_run( issueMod ) )
  return;

 f_reg_bias = reg;

 /* In the primal the bias just acquires, or loses, its quadratic coefficient
  * in the Objective; in the dual the equality constraint disappears, or comes
  * back, and the rank-one term s s^T is added to, or removed from, the
  * Hessian, which is a structural change. */
 update_abstract( remap() | ( ( AR & PrimalF ) ? eARObjective : eARAll ) ,
                  issueMod , issueAMod );

 if( issue_pmod( issueMod ) )
  Block::add_Modification( std::make_shared< SVMBlockMod >(
                            this , SVMBlockMod::eChgRegBias ) ,
                           Observer::par2chnl( issueMod ) );

 }  // end( SVMBlock::set_reg_bias )

/*--------------------------------------------------------------------------*/

void SVMBlock::set_reg_weight( double weight , ModParam issueMod ,
                               ModParam issueAMod )
{
 static const std::string _prfx = "SVMBlock::set_reg_weight: ";

 if( weight <= 0 )
  throw( std::invalid_argument( _prfx + "the weight must be positive" ) );

 if( weight == f_reg_weight )
  return;

 if( ! not_dry_run( issueMod ) )
  return;

 f_reg_weight = weight;

 // only the primal is written in terms of the weight of the regularisation
 // term, and only its Objective
 update_abstract( ( AR & PrimalF ) ? eARObjective : eARNone , issueMod ,
                  issueAMod );

 if( issue_pmod( issueMod ) )
  Block::add_Modification( std::make_shared< SVMBlockMod >(
                            this , SVMBlockMod::eChgRegWeight ) ,
                           Observer::par2chnl( issueMod ) );

 }  // end( SVMBlock::set_reg_weight )

/*--------------------------------------------------------------------------*/

void SVMBlock::chg_target( double ny , Index i , ModParam issueMod ,
                           ModParam issueAMod )
{
 if( i >= f_n )
  throw( std::invalid_argument( "SVMBlock::chg_target: invalid sample" ) );

 if( ny == v_y[ i ] )
  return;

 if( ! not_dry_run( issueMod ) )
  return;

 auto o_y = v_y;
 v_y[ i ] = ny;

 update_abstract( remap_targets( std::move( o_y ) ) , issueMod , issueAMod );

 if( issue_pmod( issueMod ) )
  Block::add_Modification( std::make_shared< SVMBlockRngdMod >(
                            this , SVMBlockMod::eChgTargets ,
                            Range( i , i + 1 ) ) ,
                           Observer::par2chnl( issueMod ) );

 }  // end( SVMBlock::chg_target )

/*--------------------------------------------------------------------------*/

void SVMBlock::chg_targets( c_doubleVec_it ny , Range rng ,
                            ModParam issueMod , ModParam issueAMod )
{
 rng.second = std::min( rng.second , f_n );
 if( rng.second <= rng.first )  // nothing to change
  return;

 if( std::equal( ny , ny + ( rng.second - rng.first ) ,
                 v_y.begin() + rng.first ) )
  return;                       // nothing changes

 if( ! not_dry_run( issueMod ) )
  return;

 auto o_y = v_y;
 std::copy( ny , ny + ( rng.second - rng.first ) , v_y.begin() + rng.first );

 update_abstract( remap_targets( std::move( o_y ) ) , issueMod , issueAMod );

 if( issue_pmod( issueMod ) )
  Block::add_Modification( std::make_shared< SVMBlockRngdMod >(
                            this , SVMBlockMod::eChgTargets , rng ) ,
                           Observer::par2chnl( issueMod ) );

 }  // end( SVMBlock::chg_targets( Range ) )

/*--------------------------------------------------------------------------*/

void SVMBlock::chg_targets( c_doubleVec_it ny , Subset && nms , bool ordered ,
                            ModParam issueMod , ModParam issueAMod )
{
 if( nms.empty() )  // nothing to change
  return;

 for( auto i : nms )
  if( i >= f_n )
   throw( std::invalid_argument( "SVMBlock::chg_targets: invalid sample" ) );

 if( ! not_dry_run( issueMod ) )
  return;

 auto o_y = v_y;

 bool changed = false;
 auto nyi = ny;
 for( auto i : nms )
  if( v_y[ i ] != *(nyi++) ) {
   changed = true;
   break;
   }

 if( ! changed )   // nothing changes
  return;

 for( auto i : nms )
  v_y[ i ] = *(ny++);

 update_abstract( remap_targets( std::move( o_y ) ) , issueMod , issueAMod );

 if( issue_pmod( issueMod ) )
  Block::add_Modification( std::make_shared< SVMBlockSbstMod >(
                            this , SVMBlockMod::eChgTargets ,
                            std::move( nms ) ) ,
                           Observer::par2chnl( issueMod ) );

 }  // end( SVMBlock::chg_targets( Subset ) )

/*--------------------------------------------------------------------------*/

void SVMBlock::copy_hyperparameters( SVMBlock * to ) const
{
 to->set_C( f_C );
 to->set_kernel( f_kernel , f_gamma , f_degree , f_coef0 );
 to->set_squared_loss( f_squared_loss );
 to->set_reg_bias( f_reg_bias );

 }  // end( SVMBlock::copy_hyperparameters )

/*--------------------------------------------------------------------------*/
/*------------- REALIGNING THE ABSTRACT REPRESENTATION ---------------------*/
/*--------------------------------------------------------------------------*/

unsigned char SVMBlock::remap( void )
{
 if( v_ds.empty() )   // there is no parametric map yet, hence no abstract
  return( eARNone );  // representation either

 auto o_ds = v_ds;
 auto o_dq = v_dq;

 set_dual_data();

 v_dcoef.clear();  // the coefficients of the model depend on the map

 if( ! AR )
  return( eARNone );

 if( ( v_ds.size() != o_ds.size() ) ||
     ( ! std::equal( v_ds.begin() , v_ds.end() , o_ds.begin() ) ) )
  return( eARAll );  // the signs are all over both formulations

 if( std::equal( v_dq.begin() , v_dq.end() , o_dq.begin() ) )
  return( eARNone );

 // the coefficients q are the linear part of the Objective of the dual and
 // the sides of the constraints of the primal
 return( ( AR & PrimalF ) ? eARSides : eARObjective );

 }  // end( SVMBlock::remap )

/*--------------------------------------------------------------------------*/

unsigned char SVMBlock::remap_targets( doubleVec && o_y )
{
 try {
  return( remap() );
  }
 catch( ... ) {
  /* The targets are only checked while the parametric map is built, hence a
   * rejected one is found once it has been written: put the previous ones
   * back, so that the SVMBlock is left exactly as it was. */
  v_y = std::move( o_y );
  set_dual_data();
  throw;
  }

 }  // end( SVMBlock::remap_targets )

/*--------------------------------------------------------------------------*/

void SVMBlock::update_abstract( unsigned char what , ModParam issueMod ,
                                ModParam issueAMod )
{
 if( ( ! AR ) || ( what == eARNone ) || ( ! not_dry_run( issueAMod ) ) )
  return;

 if( what & eARAll ) {
  rebuild_abstract( issueMod );
  return;
  }

 if( AR & HasCns ) {
  if( what & eARBounds )
   update_abstract_bounds( issueAMod );
  if( what & eARSides )
   update_abstract_sides( issueAMod );
  }

 if( ( what & eARObjective ) && ( AR & HasObj ) )
  update_abstract_objective( issueAMod );

 }  // end( SVMBlock::update_abstract )

/*--------------------------------------------------------------------------*/

void SVMBlock::update_abstract_bounds( ModParam issueAMod )
{
 if( AR & PrimalF )  // the slacks are bounded by zero, which never changes
  return;

 const double ub = get_ub();

 for( auto & bx : v_box )
  if( bx.get_rhs() != ub )
   bx.set_rhs( ub , un_ModBlock( issueAMod ) );

 }  // end( SVMBlock::update_abstract_bounds )

/*--------------------------------------------------------------------------*/

void SVMBlock::update_abstract_sides( ModParam issueAMod )
{
 if( ! ( AR & PrimalF ) )  // the dual has no side depending on the data
  return;

 for( Index k = 0 ; k < v_cons.size() ; ++k )
  if( v_cons[ k ].get_lhs() != - v_dq[ k ] )
   v_cons[ k ].set_lhs( - v_dq[ k ] , un_ModBlock( issueAMod ) );

 }  // end( SVMBlock::update_abstract_sides )

/*--------------------------------------------------------------------------*/

void SVMBlock::update_abstract_objective( ModParam issueAMod )
{
 const Index N = get_NDual();

 if( ! ( AR & PrimalF ) ) {  // the dual formulation- - - - - - - - - - - - -
                             //- - - - - - - - - - - - - - - - - - - - - - - -
  /* Only the linear coefficients and the diagonal of the Hessian are reset,
   * the off-diagonal terms depending on the Gram matrix and on the signs
   * alone: whatever changes those rebuilds the abstract representation. */
  auto & K = get_K();
  const double rb = f_reg_bias ? 1 : 0;
  const double d = f_squared_loss ? 1 / ( 2 * f_C ) : 0;

  doubleVec quad( N ) , lin( N );
  for( Index k = 0 ; k < N ; ++k ) {
   const double sk = v_ds[ k ];
   const double Kkk = K[ std::size_t( v_di[ k ] ) * f_n + v_di[ k ] ];
   lin[ k ] = - v_dq[ k ];
   quad[ k ] = - ( sk * sk * ( Kkk + rb ) + d ) / 2;
   }

  /* The diagonal and the linear coefficients of a QuadFunction are those of
   * the DQuadFunction it derives from, whence they are set through the
   * latter; the off-diagonal terms, which it adds, are left alone. */
  static_cast< DQuadFunction * >( f_obj.get_function()
   )->modify_terms( quad.begin() , lin.begin() , Range( 0 , N ) ,
                    un_ModBlock( issueAMod ) );
  return;
  }

 // the primal formulation- - - - - - - - - - - - - - - - - - - - - - - - - -
 //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

 const double rw = f_reg_weight / 2;
 const Index tot = f_m + 1 + N;

 doubleVec quad( tot ) , lin( tot , 0 );

 for( Index j = 0 ; j < f_m ; ++j )
  quad[ j ] = rw;

 quad[ f_m ] = f_reg_bias ? rw : 0;

 for( Index k = 0 ; k < N ; ++k ) {
  quad[ f_m + 1 + k ] = f_squared_loss ? f_C : 0;
  lin[ f_m + 1 + k ] = f_squared_loss ? 0 : f_C;
  }

 static_cast< DQuadFunction * >( f_obj.get_function()
  )->modify_terms( quad.begin() , lin.begin() , Range( 0 , tot ) ,
                   un_ModBlock( issueAMod ) );

 }  // end( SVMBlock::update_abstract_objective )

/*--------------------------------------------------------------------------*/

void SVMBlock::rebuild_abstract( ModParam issueMod )
{
 if( AR ) {
  const auto oAR = AR;

  delete_abstract();

  /* The same parts of the same formulation are generated anew: which one it
   * was is what the bits of AR say, so that no Configuration has to be kept
   * around for this. */
  SimpleConfiguration< int > cfg( ( oAR & PrimalF ) ? kPrimal : kWolfeDual );

  if( oAR & HasVar )
   generate_abstract_variables( & cfg );
  if( oAR & HasCns )
   generate_abstract_constraints();
  if( oAR & HasObj )
   generate_objective();
  }

 if( issue_pmod( issueMod ) && anyone_there() )
  Block::add_Modification( std::make_shared< NBModification >( this ) ,
                           Observer::par2chnl( issueMod ) );

 }  // end( SVMBlock::rebuild_abstract )


/*--------------------------------------------------------------------------*/
/*------------------------------ THE KERNEL --------------------------------*/
/*--------------------------------------------------------------------------*/

double SVMBlock::get_ub( void ) const
{
 return( f_squared_loss ? Inf< double >() : f_C );

 }  // end( SVMBlock::get_ub )

/*--------------------------------------------------------------------------*/

double SVMBlock::get_gamma( void ) const
{
 if( f_gamma > 0 )
  return( f_gamma );

 if( ! f_m )
  throw( std::logic_error( "SVMBlock::get_gamma: no data set is loaded" ) );

 /* The conventional values of gamma are derived from the data set, hence
  * they are the same at every call: they are computed once and cached, since
  * this is called for each of the O( n^2 ) entries of the Gram matrix and
  * deriving it costs O( n m ). */
 if( f_gamma_res > 0 )
  return( f_gamma_res );

 if( f_gamma == dGammaScale ) {
  // 1 / ( m * Var( X ) ), with the variance taken over all the entries
  const double sz = double( v_X.size() );
  double mean = 0;
  for( auto xi : v_X )
   mean += xi;
  mean /= sz;

  double var = 0;
  for( auto xi : v_X )
   var += ( xi - mean ) * ( xi - mean );
  var /= sz;

  if( var > 0 )
   return( f_gamma_res = 1 / ( f_m * var ) );
  }

 return( f_gamma_res = 1 / double( f_m ) );  // dGammaAuto, or a
                                             // degenerate data set

 }  // end( SVMBlock::get_gamma )

/*--------------------------------------------------------------------------*/

double SVMBlock::kernel( const double * x , const double * z ) const
{
 switch( f_kernel ) {

  case( kLinear ): {
   double d = 0;
   for( Index j = 0 ; j < f_m ; ++j )
    d += x[ j ] * z[ j ];
   return( d );
   }

  case( kPoly ): {
   double d = 0;
   for( Index j = 0 ; j < f_m ; ++j )
    d += x[ j ] * z[ j ];
   return( std::pow( get_gamma() * d + f_coef0 , f_degree ) );
   }

  case( kGaussian ): {
   double d = 0;
   for( Index j = 0 ; j < f_m ; ++j )
    d += ( x[ j ] - z[ j ] ) * ( x[ j ] - z[ j ] );
   return( std::exp( - get_gamma() * d ) );
   }

  case( kLaplacian ): {
   double d = 0;
   for( Index j = 0 ; j < f_m ; ++j )
    d += std::abs( x[ j ] - z[ j ] );
   return( std::exp( - get_gamma() * d ) );
   }

  case( kSigmoid ): {
   double d = 0;
   for( Index j = 0 ; j < f_m ; ++j )
    d += x[ j ] * z[ j ];
   return( std::tanh( get_gamma() * d + f_coef0 ) );
   }
  }

 throw( std::logic_error( "SVMBlock::kernel: unknown kernel type" ) );

 }  // end( SVMBlock::kernel )

/*--------------------------------------------------------------------------*/

SVMBlock::c_doubleVec & SVMBlock::get_K( void ) const
{
 if( v_K.size() == std::size_t( f_n ) * f_n )
  return( v_K );

 v_K.resize( std::size_t( f_n ) * f_n );

 // the value of gamma derived from the data is resolved here, and not
 // concurrently by the threads below
 if( f_kernel != kLinear )
  get_gamma();

 /* One row per iteration, each writing the upper part of its own row and the
  * corresponding part of the symmetric column, so that every entry is written
  * exactly once. The rows have very different lengths, whence the dynamic
  * scheduling; a small data set is not worth a thread, and is done here. */
 const std::size_t n = f_n;
 auto row = [ this , n ]( const long i ) {
  v_K[ std::size_t( i ) * n + i ] = kernel( Index( i ) , Index( i ) );
  for( std::size_t j = i + 1 ; j < n ; ++j ) {
   const double kij = kernel( Index( i ) , Index( j ) );
   v_K[ std::size_t( i ) * n + j ] = kij;
   v_K[ j * n + i ] = kij;
   }
  };

 const unsigned nthreads = ( f_n >= dParallelK )
  ? std::max< unsigned >( 1 , std::thread::hardware_concurrency() ) : 1;

 if( nthreads > 1 ) {
  ff::ParallelFor pf( nthreads );
  pf.parallel_for( 0 , f_n , 1 , 1 , row , nthreads );
  }
 else
  for( Index i = 0 ; i < f_n ; ++i )
   row( i );

 return( v_K );

 }  // end( SVMBlock::get_K )

/*--------------------------------------------------------------------------*/
/*---------------------- THE ABSTRACT REPRESENTATION -----------------------*/
/*--------------------------------------------------------------------------*/

void SVMBlock::generate_abstract_variables( Configuration * stvv )
{
 static const std::string _prfx =
                                "SVMBlock::generate_abstract_variables: ";

 if( AR & HasVar )  // the Variable are there already
  return;           // nothing to do

 check_data();

 /* Which problem the abstract representation encodes is a Configuration
  * matter: it is not part of the training problem, hence it is not part of
  * the data of the SVMBlock. */
 int wf = kWolfeDual;

 if( ( ! stvv ) && f_BlockConfig )
  stvv = f_BlockConfig->f_static_variables_Configuration;

 if( auto sci = dynamic_cast< SimpleConfiguration< int > * >( stvv ) )
  wf = sci->f_value;

 if( ( wf != kWolfeDual ) && ( wf != kPrimal ) )
  throw( std::invalid_argument( _prfx + "unknown problem" ) );

 if( ( wf == kPrimal ) && ( f_kernel != kLinear ) )
  throw( std::invalid_argument( _prfx + "the primal is only available for "
                                "the linear kernel" ) );

 const Index N = get_NDual();

 if( wf == kWolfeDual ) {  // the Wolfe dual - - - - - - - - - - - - - - - - -
                           //- - - - - - - - - - - - - - - - - - - - - - - - -
  v_alpha_var.resize( N );
  for( auto & ak : v_alpha_var )
   ak.set_type( ColVariable::kContinuous );

  add_static_variable( v_alpha_var , "alpha" );
  }
 else {                    // the training problem itself- - - - - - - - - - -
                           //- - - - - - - - - - - - - - - - - - - - - - - - -
  AR |= PrimalF;

  v_w.resize( f_m );
  for( auto & wj : v_w )
   wj.set_type( ColVariable::kContinuous );
  add_static_variable( v_w , "w" );

  f_b_var.set_type( ColVariable::kContinuous );
  add_static_variable( f_b_var , "b" );

  v_xi.resize( N );
  for( auto & xk : v_xi )
   xk.set_type( ColVariable::kContinuous );
  add_static_variable( v_xi , "xi" );
  }

 AR |= HasVar;

 }  // end( SVMBlock::generate_abstract_variables )


/*--------------------------------------------------------------------------*/

void SVMBlock::generate_abstract_constraints( Configuration * stcc )
{
 static const std::string _prfx =
                              "SVMBlock::generate_abstract_constraints: ";

 if( AR & HasCns )  // the Constraint are there already
  return;           // nothing to do

 if( ! ( AR & HasVar ) )
  throw( std::logic_error( _prfx + "generate_abstract_variables not called"
                           ) );

 const Index N = get_NDual();

 if( ! ( AR & PrimalF ) ) {  // the dual formulation- - - - - - - - - - - - -
                             //- - - - - - - - - - - - - - - - - - - - - - - -
  const double ub = get_ub();

  v_box.resize( N );
  for( Index k = 0 ; k < N ; ++k ) {
   v_box[ k ].set_variable( & v_alpha_var[ k ] );
   v_box[ k ].set_rhs( ub );
   }

  add_static_constraint( v_box , "box" );

  if( ! f_reg_bias ) {  // s^T alpha = 0
   v_coeff_pair coeffs( N );
   for( Index k = 0 ; k < N ; ++k )
    coeffs[ k ] = std::make_pair( & v_alpha_var[ k ] , v_ds[ k ] );

   f_eq.set_both( 0 );
   f_eq.set_function( new LinearFunction( std::move( coeffs ) , 0 ) );

   add_static_constraint( f_eq , "eq" );
   }
  }
 else {                      // the primal formulation - - - - - - - - - - - -
                             //- - - - - - - - - - - - - - - - - - - - - - - -
  v_xi_box.resize( N );
  for( Index k = 0 ; k < N ; ++k )
   v_xi_box[ k ].set_variable( & v_xi[ k ] );

  add_static_constraint( v_xi_box , "xibox" );

  // s_k ( < w , x_{ i( k ) } > + b ) + xi_k >= r_k
  v_cons.resize( N );
  for( Index k = 0 ; k < N ; ++k ) {
   const double sk = v_ds[ k ];
   const double * xi = get_x( v_di[ k ] );

   v_coeff_pair coeffs( f_m + 2 );
   for( Index j = 0 ; j < f_m ; ++j )
    coeffs[ j ] = std::make_pair( & v_w[ j ] , sk * xi[ j ] );

   coeffs[ f_m ] = std::make_pair( & f_b_var , sk );
   coeffs[ f_m + 1 ] = std::make_pair( & v_xi[ k ] , double( 1 ) );

   v_cons[ k ].set_lhs( - v_dq[ k ] );
   v_cons[ k ].set_rhs( Inf< RowConstraint::RHSValue >() );
   v_cons[ k ].set_function( new LinearFunction( std::move( coeffs ) , 0 ) );
   }

  add_static_constraint( v_cons , "cons" );
  }

 AR |= HasCns;

 }  // end( SVMBlock::generate_abstract_constraints )

/*--------------------------------------------------------------------------*/

void SVMBlock::generate_objective( Configuration * objc )
{
 static const std::string _prfx = "SVMBlock::generate_objective: ";

 if( AR & HasObj )  // the Objective is there already
  return;           // nothing to do

 if( ! ( AR & HasVar ) )
  throw( std::logic_error( _prfx + "generate_abstract_variables not called"
                           ) );

 const Index N = get_NDual();

 if( ! ( AR & PrimalF ) ) {  // the dual formulation- - - - - - - - - - - - -
                             //- - - - - - - - - - - - - - - - - - - - - - - -
  /* max - q^T alpha - 1/2 alpha^T Q alpha, i.e., the Wolfe dual as it is
   * customarily written, with the diagonal of Q halved into the quadratic
   * coefficients of the QuadFunction and the strictly lower triangle of Q
   * into its off-diagonal terms, everything negated since the objective is
   * maximised. Writing it this way, rather than as the minimisation of the
   * opposite, is what makes the *value* of the Objective the same in all the
   * formulations, strong duality holding since the training problem is
   * convex: a Solver on the dual and one on either primal then agree on a
   * number, instead of on two numbers that happen to be opposite. */
  auto & K = get_K();
  const double rb = f_reg_bias ? 1 : 0;
  const double d = f_squared_loss ? 1 / ( 2 * f_C ) : 0;

  v_coeff_triple triples( N );
  for( Index k = 0 ; k < N ; ++k ) {
   const double sk = v_ds[ k ];
   const double Kkk = K[ std::size_t( v_di[ k ] ) * f_n + v_di[ k ] ];
   triples[ k ] = std::make_tuple( & v_alpha_var[ k ] , - v_dq[ k ] ,
                                   - ( sk * sk * ( Kkk + rb ) + d ) / 2 );
   }

  v_off_diag_term off_diag;
  off_diag.reserve( ( std::size_t( N ) * ( N - 1 ) ) / 2 );
  for( Index k = 1 ; k < N ; ++k ) {
   const double sk = v_ds[ k ];
   const std::size_t ik = std::size_t( v_di[ k ] ) * f_n;
   for( Index l = 0 ; l < k ; ++l ) {
    const double Qkl = sk * v_ds[ l ] * ( K[ ik + v_di[ l ] ] + rb );
    if( Qkl )
     off_diag.emplace_back( k , l , - Qkl );
    }
   }

  f_obj.set_function( new QuadFunction( std::move( triples ) ,
                                        std::move( off_diag ) , 0 ) ,
                      eNoMod );

  f_obj.set_sense( Objective::eMax , eNoMod );
  set_objective( & f_obj , eNoMod );

  AR |= HasObj;
  return;
  }
 else {                      // the primal formulation - - - - - - - - - - - -
                             //- - - - - - - - - - - - - - - - - - - - - - - -
  // min rw/2 || w ||^2 [ + rw/2 b^2 ] + C sum_k xi_k^p, with rw the weight of
  // the regularisation term, which is 1 unless this is the sub-Block of a
  // decomposition, in which case the term is split among the sub-Block
  const double rw = f_reg_weight / 2;

  v_coeff_triple triples( f_m + 1 + N );

  for( Index j = 0 ; j < f_m ; ++j )
   triples[ j ] = std::make_tuple( & v_w[ j ] , double( 0 ) , rw );

  triples[ f_m ] = std::make_tuple( & f_b_var , double( 0 ) ,
                                    f_reg_bias ? rw : double( 0 ) );

  for( Index k = 0 ; k < N ; ++k )
   triples[ f_m + 1 + k ] =
    std::make_tuple( & v_xi[ k ] , f_squared_loss ? double( 0 ) : f_C ,
                     f_squared_loss ? f_C : double( 0 ) );

  f_obj.set_function( new DQuadFunction( std::move( triples ) , 0 ) ,
                      eNoMod );
  }

 f_obj.set_sense( Objective::eMin , eNoMod );
 set_objective( & f_obj , eNoMod );

 AR |= HasObj;

 }  // end( SVMBlock::generate_objective )

/*--------------------------------------------------------------------------*/
/*------------------------- THE TRAINED MODEL ------------------------------*/
/*--------------------------------------------------------------------------*/

void SVMBlock::set_dual_solution( doubleVec && alpha , double b )
{
 if( alpha.size() != get_NDual() )
  throw( std::invalid_argument( "SVMBlock::set_dual_solution: alpha has "
                                "wrong size" ) );

 v_alpha = std::move( alpha );
 v_w_sol.clear();
 f_b = b;
 v_dcoef.clear();

 }  // end( SVMBlock::set_dual_solution )

/*--------------------------------------------------------------------------*/

void SVMBlock::set_primal_solution( doubleVec && w , double b )
{
 if( w.size() != f_m )
  throw( std::invalid_argument( "SVMBlock::set_primal_solution: w has wrong "
                                "size" ) );

 v_w_sol = std::move( w );
 v_alpha.assign( get_NDual() , 0 );
 f_b = b;
 v_dcoef.clear();

 }  // end( SVMBlock::set_primal_solution )

/*--------------------------------------------------------------------------*/

void SVMBlock::get_solution_from_abstract( void )
{
 static const std::string _prfx = "SVMBlock::get_solution_from_abstract: ";

 if( ! ( AR & HasVar ) )
  throw( std::logic_error( _prfx + "no abstract representation" ) );

 v_dcoef.clear();

 if( ! ( AR & PrimalF ) ) {  // the dual formulation- - - - - - - - - - - - -
                             //- - - - - - - - - - - - - - - - - - - - - - - -
  v_alpha.resize( v_alpha_var.size() );
  for( Index k = 0 ; k < v_alpha_var.size() ; ++k )
   v_alpha[ k ] = v_alpha_var[ k ].get_value();

  v_w_sol.clear();
  compute_bias();
  }
 else {                      // the primal formulation - - - - - - - - - - - -
                             //- - - - - - - - - - - - - - - - - - - - - - - -
  v_w_sol.resize( f_m );
  for( Index j = 0 ; j < f_m ; ++j )
   v_w_sol[ j ] = v_w[ j ].get_value();

  v_alpha.assign( get_NDual() , 0 );
  f_b = f_b_var.get_value();
  }

 }  // end( SVMBlock::get_solution_from_abstract )

/*--------------------------------------------------------------------------*/

void SVMBlock::set_solution_in_abstract( void )
{
 if( ! ( AR & HasVar ) )  // there is nothing to write into
  return;

 if( ! ( AR & PrimalF ) ) {  // the dual formulation- - - - - - - - - - - - -
                                           //- - - - - - - - - - - - - - - - -
  for( Index k = 0 ; k < v_alpha_var.size() ; ++k )
   v_alpha_var[ k ].set_value( k < v_alpha.size() ? v_alpha[ k ] : 0 );

  return;
  }

 // both primal formulations need the weights, which only exist for the
 // linear kernel, which is also the only one they exist for
 auto w = get_w();

 // the primal formulation - - - - - - - - - - - - - - - - - - - - - - - - - -
 //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

 for( Index j = 0 ; j < f_m ; ++j )
  v_w[ j ].set_value( w[ j ] );

 f_b_var.set_value( f_b );

 // the slacks are the smallest values that make the model feasible, which is
 // what they are worth at any optimal solution of the primal
 for( Index k = 0 ; k < v_xi.size() ; ++k ) {
  const double * xi = get_x( v_di[ k ] );

  double f = f_b;
  for( Index j = 0 ; j < f_m ; ++j )
   f += w[ j ] * xi[ j ];

  v_xi[ k ].set_value( std::max( double( 0 ) ,
                                 - v_dq[ k ] - v_ds[ k ] * f ) );
  }

 }  // end( SVMBlock::set_solution_in_abstract )

/*--------------------------------------------------------------------------*/

SVMBlock::c_doubleVec & SVMBlock::get_dual_coefficients( void ) const
{
 if( v_dcoef.size() == f_n )
  return( v_dcoef );

 v_dcoef.assign( f_n , 0 );
 for( Index k = 0 ; k < v_alpha.size() ; ++k )
  v_dcoef[ v_di[ k ] ] += v_ds[ k ] * v_alpha[ k ];

 return( v_dcoef );

 }  // end( SVMBlock::get_dual_coefficients )

/*--------------------------------------------------------------------------*/

SVMBlock::doubleVec SVMBlock::get_w( void ) const
{
 if( f_kernel != kLinear )
  throw( std::logic_error( "SVMBlock::get_w: the weight vector only exists "
                           "for the linear kernel" ) );

 if( ! v_w_sol.empty() )
  return( v_w_sol );

 auto & c = get_dual_coefficients();

 doubleVec w( f_m , 0 );
 for( Index i = 0 ; i < f_n ; ++i ) {
  if( ! c[ i ] )
   continue;
  const double * xi = get_x( i );
  for( Index j = 0 ; j < f_m ; ++j )
   w[ j ] += c[ i ] * xi[ j ];
  }

 return( w );

 }  // end( SVMBlock::get_w )

/*--------------------------------------------------------------------------*/

double SVMBlock::decision_function( const double * x ) const
{
 double d = f_b;

 if( ! v_w_sol.empty() ) {  // the model came out of the primal
  for( Index j = 0 ; j < f_m ; ++j )
   d += v_w_sol[ j ] * x[ j ];
  return( d );
  }

 auto & c = get_dual_coefficients();

 for( Index i = 0 ; i < f_n ; ++i )
  if( c[ i ] )
   d += c[ i ] * kernel( get_x( i ) , x );

 return( d );

 }  // end( SVMBlock::decision_function )

/*--------------------------------------------------------------------------*/

double SVMBlock::dual_objective( c_doubleVec & alpha ) const
{
 if( alpha.size() != get_NDual() )
  throw( std::invalid_argument( "SVMBlock::dual_objective: alpha has wrong "
                                "size" ) );

 auto & K = get_K();
 const double rb = f_reg_bias ? 1 : 0;
 const double d = f_squared_loss ? 1 / ( 2 * f_C ) : 0;
 const Index N = alpha.size();

 // sum_i c_i K( x_i , . ) is the only nondiagonal part of alpha^T Q alpha
 doubleVec c( f_n , 0 );
 for( Index k = 0 ; k < N ; ++k )
  c[ v_di[ k ] ] += v_ds[ k ] * alpha[ k ];

 double quad = 0;
 for( Index i = 0 ; i < f_n ; ++i ) {
  if( ! c[ i ] )
   continue;
  const std::size_t ii = std::size_t( i ) * f_n;
  for( Index j = 0 ; j < f_n ; ++j )
   quad += c[ i ] * c[ j ] * K[ ii + j ];
  }

 if( rb ) {  // the rank-one term ( s^T alpha )^2
  double sa = 0;
  for( Index k = 0 ; k < N ; ++k )
   sa += v_ds[ k ] * alpha[ k ];
  quad += sa * sa;
  }

 double lin = 0;
 for( Index k = 0 ; k < N ; ++k ) {
  lin += v_dq[ k ] * alpha[ k ];
  if( d )
   quad += d * alpha[ k ] * alpha[ k ];
  }

 // the Wolfe dual is maximised, hence the opposite of the quadratic form
 return( - ( quad / 2 + lin ) );

 }  // end( SVMBlock::dual_objective )

/*--------------------------------------------------------------------------*/

void SVMBlock::compute_bias( void )
{
 v_dcoef.clear();

 if( f_reg_bias ) {  // the bias is just one more weight
  double b = 0;
  for( Index k = 0 ; k < v_alpha.size() ; ++k )
   b += v_ds[ k ] * v_alpha[ k ];
  f_b = b;
  return;
  }

 auto & c = get_dual_coefficients();
 auto & K = get_K();

 const double ub = get_ub();
 const double lo = dBndEps * ( f_C > 1 ? f_C : 1 );
 const double hi = ( ub < Inf< double >() ) ? ub - lo : Inf< double >();

 double sum = 0;
 Index cnt = 0;

 for( Index k = 0 ; k < v_alpha.size() ; ++k ) {
  const double ak = v_alpha[ k ];
  if( ( ak <= lo ) || ( ak >= hi ) )  // not a free support vector
   continue;

  const Index i = v_di[ k ];
  const std::size_t ii = std::size_t( i ) * f_n;

  double g = 0;
  for( Index j = 0 ; j < f_n ; ++j )
   if( c[ j ] )
    g += c[ j ] * K[ ii + j ];

  const double xik = f_squared_loss ? ak / ( 2 * f_C ) : 0;
  sum += v_ds[ k ] * ( - v_dq[ k ] - xik ) - g;
  ++cnt;
  }

 if( cnt )
  f_b = sum / cnt;

 }  // end( SVMBlock::compute_bias )

/*--------------------------------------------------------------------------*/
/*--------------------------- OTHER METHODS --------------------------------*/
/*--------------------------------------------------------------------------*/

void SVMBlock::guts_of_destructor( void )
{
 delete_abstract();

 v_K.clear();
 v_dcoef.clear();
 f_gamma_res = 0;

 }  // end( SVMBlock::guts_of_destructor )

/*--------------------------------------------------------------------------*/

void SVMBlock::delete_abstract( void )
{
 /* clear() all Constraint so that they do not bother to un-register
  * themselves from Variable that are going to be deleted anyway, then delete
  * the whole abstract representation. The cached quantities, which do not
  * depend on it, are left alone: this is also called to rebuild the abstract
  * representation, in which case recomputing the Gram matrix would be a
  * needless O( n^2 m ). */

 for( auto & cnst : v_cons )
  cnst.clear();
 for( auto & cnst : v_xi_box )
  cnst.clear();
 f_eq.clear();
 for( auto & cnst : v_box )
  cnst.clear();

 f_obj.clear();

 v_cons.clear();
 v_xi_box.clear();
 v_box.clear();

 v_xi.clear();
 v_w.clear();
 v_alpha_var.clear();

 /* The bias of the primal is the only Variable of the SVMBlock that does not
  * live in a container, hence the only one that is not destroyed here: since
  * the Constraint and the Objective it was active in have been destroyed
  * without un-registering, it would be left pointing at them, whence it is
  * replaced by a fresh copy of itself, which the copy constructor of
  * ColVariable makes active in nothing. */
 f_b_var = ColVariable( f_b_var );

 reset_static_constraints();
 reset_static_variables();
 reset_objective();

 AR = 0;

 }  // end( SVMBlock::delete_abstract )

/*--------------------------------------------------------------------------*/

void SVMBlock::print( std::ostream & output , char vlvl ) const
{
 output << classname() << " with " << f_n << " samples of " << f_m
        << " features, C = " << f_C << ", kernel = " << f_kernel;

 if( f_squared_loss )
  output << ", squared loss";
 if( f_reg_bias )
  output << ", regularized bias";

 output << std::endl;

 if( ! vlvl )
  return;

 output << std::setprecision( 8 );
 for( Index i = 0 ; i < f_n ; ++i ) {
  const double * xi = get_x( i );
  for( Index j = 0 ; j < f_m ; ++j )
   output << xi[ j ] << "\t";
  output << v_y[ i ] << std::endl;
  }

 }  // end( SVMBlock::print )

/*--------------------------------------------------------------------------*/
/*--------------------- FUNCTIONS OF THE SVMBlock GROUP --------------------*/
/*--------------------------------------------------------------------------*/

Block * SMSpp_di_unipi_it::make_consensus_Block( const SVMBlock * svm ,
                                                Block::Index P )
{
 static const std::string _prfx = "make_consensus_Block: ";

 if( ! svm )
  throw( std::invalid_argument( _prfx + "no SVMBlock given" ) );

 if( ! P )
  throw( std::invalid_argument( _prfx + "the chunks must be at least one" ) );

 if( svm->get_kernel_type() != SVMBlock::kLinear )
  throw( std::invalid_argument( _prfx + "only the linear kernel has the "
                                "explicit feature map the primal needs" ) );

 const Block::Index n = svm->get_NSamples();
 const Block::Index m = svm->get_NFeatures();

 if( P > n )
  throw( std::invalid_argument( _prfx + "more chunks than samples" ) );

 /* The samples are dealt out to the chunks round-robin after having been
  * sorted by target, so that consecutive samples in the order end up in
  * different chunks: for a classification problem this means that each chunk
  * gets samples of both classes as long as there are at least P of the least
  * numerous one, which is what keeps its subproblem bounded. */
 std::vector< Block::Index > order( n );
 std::iota( order.begin() , order.end() , Block::Index( 0 ) );

 auto & y = svm->get_y();
 std::stable_sort( order.begin() , order.end() ,
                   [ &y ]( Block::Index i , Block::Index j ) {
                    return( y[ i ] < y[ j ] );
                    } );

 std::vector< std::vector< Block::Index > > chunk( P );
 for( Block::Index t = 0 ; t < n ; ++t )
  chunk[ t % P ].push_back( order[ t ] );

 // the father, which has no Variable of its own- - - - - - - - - - - - - - -

 auto father = new AbstractBlock();

 // one sub-Block per chunk - - - - - - - - - - - - - - - - - - - - - - - - -

 for( Block::Index p = 0 ; p < P ; ++p ) {
  auto sub = dynamic_cast< SVMBlock * >(
                       Block::new_Block( svm->classname() , father ) );
  if( ! sub ) {
   delete father;
   throw( std::logic_error( _prfx + svm->classname() +
                            " is not in the Block factory" ) );
   }

  svm->copy_hyperparameters( sub );

  /* The regularisation term is *split* among the chunks, rather than being
   * left in one designated chunk, so that every subproblem stays strongly
   * convex, hence bounded, whatever the multipliers. */
  sub->set_reg_weight( 1 / double( P ) );

  SVMBlock::doubleVec X( chunk[ p ].size() * m ) , yp( chunk[ p ].size() );

  for( Block::Index t = 0 ; t < chunk[ p ].size() ; ++t ) {
   std::copy_n( svm->get_x( chunk[ p ][ t ] ) , m , X.begin() + t * m );
   yp[ t ] = y[ chunk[ p ][ t ] ];
   }

  sub->load( chunk[ p ].size() , m , std::move( X ) , std::move( yp ) );

  /* A chunk whose dual signs are all equal has an unbounded Lagrangian
   * subproblem in its bias, since the latter then only appears linearly and
   * moving it in the right direction relaxes all the constraints at once.
   * Regularising the bias makes the subproblem strongly convex in it, hence
   * bounded, so only the other case has to be refused. */
  if( ! svm->get_reg_bias() ) {
   auto & sg = sub->get_dual_signs();
   if( std::all_of( sg.begin() , sg.end() ,
                    [ &sg ]( double sk ) { return( sk == sg[ 0 ] ); } ) ) {
    delete father;
    throw( std::invalid_argument(
     _prfx + "chunk " + std::to_string( p ) + " has samples of one class "
     "only, whose Lagrangian subproblem is unbounded in the bias: either "
     "use fewer chunks or regularise the bias" ) );
    }
   }

  SimpleConfiguration< int > primal( SVMBlock::kPrimal );
  sub->generate_abstract_variables( & primal );
  sub->generate_abstract_constraints();
  sub->generate_objective();

  father->add_nested_Block( sub );
  }

 // the consensus constraints, the only ones linking the sub-Block - - - - - -

 auto link = new std::vector< FRowConstraint >( ( P - 1 ) * ( m + 1 ) );

 auto lnk = link->begin();
 for( Block::Index p = 0 ; p + 1 < P ; ++p ) {
  auto sp = father->get_nested_Block( p );
  auto sq = father->get_nested_Block( p + 1 );

  auto wp = sp->get_static_variable_v< ColVariable >( "w" );
  auto wq = sq->get_static_variable_v< ColVariable >( "w" );
  auto bp = sp->get_static_variable< ColVariable >( "b" );
  auto bq = sq->get_static_variable< ColVariable >( "b" );

  for( Block::Index j = 0 ; j < m ; ++j , ++lnk ) {
   LinearFunction::v_coeff_pair coeffs( 2 );
   coeffs[ 0 ] = std::make_pair( & (*wp)[ j ] , double( 1 ) );
   coeffs[ 1 ] = std::make_pair( & (*wq)[ j ] , double( -1 ) );
   lnk->set_both( 0 );
   lnk->set_function( new LinearFunction( std::move( coeffs ) , 0 ) );
   }

  LinearFunction::v_coeff_pair coeffs( 2 );
  coeffs[ 0 ] = std::make_pair( bp , double( 1 ) );
  coeffs[ 1 ] = std::make_pair( bq , double( -1 ) );
  lnk->set_both( 0 );
  lnk->set_function( new LinearFunction( std::move( coeffs ) , 0 ) );
  ++lnk;
  }

 if( P > 1 )
  father->add_static_constraint( *link , "link" );
 else
  delete link;

 /* The Objective is all in the sub-Block, the father having no Variable at
  * all; yet an *empty* one is set, because a Solver flattening the whole tree
  * needs it to know the sense of the problem, while a Lagrangian one is fine
  * with it as long as it depends on no Variable. */
 auto obj = new FRealObjective();
 obj->set_function( new LinearFunction() );
 obj->set_sense( Objective::eMin , eNoMod );
 father->set_objective( obj );

 return( father );

 }  // end( make_consensus_Block )

/*--------------------------------------------------------------------------*/
/*------------------- METHODS OF THE CLASS SVMBlockSolution ----------------*/
/*--------------------------------------------------------------------------*/

void SVMBlockSolution::deserialize( const netCDF::NcGroup & group )
{
 auto read = []( const netCDF::NcGroup & group , const std::string & dim ,
                 const std::string & var , SVMBlock::doubleVec & data ) {
  auto nc_dim = group.getDim( dim );
  auto nc_var = group.getVar( var );

  if( nc_dim.isNull() || nc_var.isNull() ) {
   data.clear();
   return;
   }

  data.resize( nc_dim.getSize() );
  nc_var.getVar( data.data() );
  };

 read( group , "NMultipliers" , "Multipliers" , v_alpha );
 read( group , "NFeatures" , "Weights" , v_w );

 f_b = 0;
 ::deserialize( group , f_b , "Bias" , true );

 }  // end( SVMBlockSolution::deserialize )

/*--------------------------------------------------------------------------*/

void SVMBlockSolution::serialize( netCDF::NcGroup & group ) const
{
 // always call the method of the base class first
 Solution::serialize( group );

 const std::vector< std::size_t > start = { 0 };

 if( ! v_alpha.empty() ) {
  auto dim = group.addDim( "NMultipliers" , v_alpha.size() );
  const std::vector< std::size_t > count = { v_alpha.size() };
  ( group.addVar( "Multipliers" , netCDF::NcDouble() , dim ) ).putVar(
                                           start , count , v_alpha.data() );
  }

 if( ! v_w.empty() ) {
  auto dim = group.addDim( "NFeatures" , v_w.size() );
  const std::vector< std::size_t > count = { v_w.size() };
  ( group.addVar( "Weights" , netCDF::NcDouble() , dim ) ).putVar(
                                               start , count , v_w.data() );
  }

 ::serialize( group , "Bias" , netCDF::NcDouble() , f_b );

 }  // end( SVMBlockSolution::serialize )

/*--------------------------------------------------------------------------*/

void SVMBlockSolution::read( const Block * block )
{
 auto svm = dynamic_cast< const SVMBlock * >( block );
 if( ! svm )
  throw( std::invalid_argument( "SVMBlockSolution::read: block is not a "
                                "SVMBlock" ) );

 /* The model may well be in the abstract representation rather than in the
  * physical one, which is the case whenever the SVMBlock has just been solved
  * by a Solver working on the former: the latter is therefore refreshed out
  * of it first, so that what is saved here is the model that has just been
  * found, whoever has found it. The const_cast is what refreshing a cached
  * view of the very same information out of a const object costs. */
 if( svm->AR & SVMBlock::HasVar )
  const_cast< SVMBlock * >( svm )->get_solution_from_abstract();

 v_alpha = svm->v_alpha;
 v_w = svm->v_w_sol;
 f_b = svm->f_b;

 }  // end( SVMBlockSolution::read )

/*--------------------------------------------------------------------------*/

void SVMBlockSolution::write( Block * block )
{
 auto svm = dynamic_cast< SVMBlock * >( block );
 if( ! svm )
  throw( std::invalid_argument( "SVMBlockSolution::write: block is not a "
                                "SVMBlock" ) );

 /* The model is written the way it was obtained: out of the weights if it
  * comes from a primal, where the multipliers are unknown, out of the
  * multipliers otherwise. */
 if( ! v_w.empty() )
  svm->set_primal_solution( SVMBlock::doubleVec( v_w ) , f_b );
 else
  if( ! v_alpha.empty() )
   svm->set_dual_solution( SVMBlock::doubleVec( v_alpha ) , f_b );

 svm->set_solution_in_abstract();

 }  // end( SVMBlockSolution::write )

/*--------------------------------------------------------------------------*/

SVMBlockSolution * SVMBlockSolution::scale( double factor ) const
{
 auto sol = SVMBlockSolution::clone( true );

 for( Block::Index i = 0 ; i < v_alpha.size() ; ++i )
  sol->v_alpha[ i ] = v_alpha[ i ] * factor;

 for( Block::Index i = 0 ; i < v_w.size() ; ++i )
  sol->v_w[ i ] = v_w[ i ] * factor;

 sol->f_b = f_b * factor;

 return( sol );

 }  // end( SVMBlockSolution::scale )

/*--------------------------------------------------------------------------*/

void SVMBlockSolution::sum( const Solution * solution , double multiplier )
{
 auto sol = dynamic_cast< const SVMBlockSolution * >( solution );
 if( ! sol )
  throw( std::invalid_argument( "SVMBlockSolution::sum: solution is not a "
                                "SVMBlockSolution" ) );

 if( ! v_alpha.empty() ) {
  if( v_alpha.size() != sol->v_alpha.size() )
   throw( std::invalid_argument( "SVMBlockSolution::sum: incompatible "
                                 "number of multipliers" ) );

  for( Block::Index i = 0 ; i < v_alpha.size() ; ++i )
   v_alpha[ i ] += sol->v_alpha[ i ] * multiplier;
  }

 if( ! v_w.empty() ) {
  if( v_w.size() != sol->v_w.size() )
   throw( std::invalid_argument( "SVMBlockSolution::sum: incompatible "
                                 "number of weights" ) );

  for( Block::Index i = 0 ; i < v_w.size() ; ++i )
   v_w[ i ] += sol->v_w[ i ] * multiplier;
  }

 f_b += sol->f_b * multiplier;

 }  // end( SVMBlockSolution::sum )

/*--------------------------------------------------------------------------*/

SVMBlockSolution * SVMBlockSolution::clone( bool empty ) const
{
 auto sol = new SVMBlockSolution();

 if( empty ) {
  sol->v_alpha.resize( v_alpha.size() );
  sol->v_w.resize( v_w.size() );
  }
 else {
  sol->v_alpha = v_alpha;
  sol->v_w = v_w;
  sol->f_b = f_b;
  }

 return( sol );

 }  // end( SVMBlockSolution::clone )

/*--------------------------------------------------------------------------*/
/*------------------------- End File SVMBlock.cpp --------------------------*/
/*--------------------------------------------------------------------------*/
