/*--------------------------------------------------------------------------*/
/*--------------------------- File SVMBlock.cppm ---------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Interface of the C++20 module smspp.svmblock, which exports every name that
 * the headers of SVMBlock declare at namespace scope, together with the
 * modules it depends on; a translation unit can then write
 *
 *     import smspp.svmblock;
 *
 * in place of the #include of the headers. The macros (e.g., those of the
 * factory in SMSFactory.h) are not exported by a module, so a file that uses
 * them still includes the header that defines them.
 *
 * This file is written by SMS++/cmake/smspp_module_exports.py, which reads
 * the names from the AST of the headers: it is not to be edited by hand.
 *
 * \author Donato Meoli \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 *
 * \copyright &copy; by Donato Meoli
 */
/*--------------------------------------------------------------------------*/
/*------------------------- GLOBAL MODULE FRAGMENT -------------------------*/
/*--------------------------------------------------------------------------*/

module;

#include "LIBLINEARSolver.h"
#include "LIBSVMSolver.h"
#include "SMOSolver.h"
#include "SVCBlock.h"
#include "SVMBlock.h"
#include "SVMBlockArch.h"
#include "SVRBlock.h"

/*--------------------------------------------------------------------------*/
/*----------------------------- MODULE PURVIEW -----------------------------*/
/*--------------------------------------------------------------------------*/

export module smspp.svmblock;

export import smspp;

export namespace SMSpp_di_unipi_it {

 // LIBLINEARSolver.h
 using SMSpp_di_unipi_it::LIBLINEARSolver;

 // LIBSVMSolver.h
 using SMSpp_di_unipi_it::LIBSVMSolver;

 // SMOSolver.h
 using SMSpp_di_unipi_it::SMOSolver;

 // SVCBlock.h
 using SMSpp_di_unipi_it::SVCBlock;

 // SVMBlock.h
 using SMSpp_di_unipi_it::SVMBlock;
 using SMSpp_di_unipi_it::SVMBlockMod;
 using SMSpp_di_unipi_it::SVMBlockRngdMod;
 using SMSpp_di_unipi_it::SVMBlockSbstMod;
 using SMSpp_di_unipi_it::SVMBlockSolution;

 // SVRBlock.h
 using SMSpp_di_unipi_it::SVRBlock;
}

/*--------------------------------------------------------------------------*/
/*------------------------- End File SVMBlock.cppm -------------------------*/
/*--------------------------------------------------------------------------*/
