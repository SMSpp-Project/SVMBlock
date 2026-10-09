/*--------------------------------------------------------------------------*/
/*---------------------------- File svm_arch.cpp ---------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 *
 * Measures the few properties of the machine that the defaults of SVMBlock
 * depend on, and writes them into the header SVMBlockArch.h, which is
 * generated at the first compilation and is therefore the machine's own: the
 * memory it has, how many cores it has, what one entry of a dense inner
 * product costs, what one entry of the merge of two lists of nonzeroes costs,
 * and what starting a thread costs. The rules that read them are in SVMBlock
 * [see SVMBlock::set_sparse_density(), set_K_memory() and get_K()], so that
 * what is measured here are the parameters of the architecture and not the
 * decisions themselves.
 *
 * The header is not part of the sources, a value measured on one machine
 * saying nothing about another; whoever compiles without it gets the
 * conservative defaults that SVMBlock carries.
 *
 * \author Donato Meoli \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 *
 * \copyright &copy; by Donato Meoli
 */
/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#ifdef _WIN32
 #include <windows.h>
#else
 #include <unistd.h>
#endif

/*--------------------------------------------------------------------------*/
/*------------------------------ FUNCTIONS ---------------------------------*/
/*--------------------------------------------------------------------------*/
/// how many bytes of memory the machine has, zero if it will not say

static double total_memory( void )
{
#ifdef _WIN32
 MEMORYSTATUSEX st;
 st.dwLength = sizeof( st );
 if( GlobalMemoryStatusEx( & st ) )
  return( double( st.ullTotalPhys ) );
 return( 0 );
#else
 const long pages = sysconf( _SC_PHYS_PAGES );
 const long page = sysconf( _SC_PAGE_SIZE );
 if( ( pages > 0 ) && ( page > 0 ) )
  return( double( pages ) * double( page ) );
 return( 0 );
#endif
 }

/*--------------------------------------------------------------------------*/
/// the seconds that \p f takes, repeated until they are worth reading
/** Runs \p f until the total time is a hundredth of a second, which is enough
 * above the resolution of the clock, and returns the seconds one run takes.
 * The result of \p f is accumulated into \p sink, so that no optimizer is
 * free to throw the whole thing away. */

template< class F >
static double timed( F && f , volatile double & sink )
{
 unsigned long reps = 1;

 for( ; ; reps *= 2 ) {
  const auto start = std::chrono::steady_clock::now();
  double acc = 0;
  for( unsigned long r = 0 ; r < reps ; ++r )
   acc += f();
  const double seconds = std::chrono::duration< double >(
                          std::chrono::steady_clock::now() - start ).count();
  sink = sink + acc;

  if( seconds > 0.01 )
   return( seconds / double( reps ) );

  if( reps > ( 1UL << 40 ) )   // the clock is not going to say anything
   return( 0 );
  }
 }

/*--------------------------------------------------------------------------*/
/*--------------------------------- MAIN -----------------------------------*/
/*--------------------------------------------------------------------------*/

int main( int argc , char ** argv )
{
 const std::string path = ( argc > 1 ) ? argv[ 1 ] : "../include";

 volatile double sink = 0;

 /* The two ways of computing the same inner product: reading the two samples
  * dense, which costs m entries whatever they hold, and merging the two lists
  * of their nonzeroes, which costs the nonzeroes but reads an index per entry
  * and branches at each of them. Which of the two wins is a property of the
  * machine and of how full the samples are, so the crossing point is measured
  * rather than derived: at each density a data set of that density is built
  * and the kernel of random pairs of its samples is computed both ways, the
  * data set being large enough not to sit in the cache, as the one of a
  * training is not. The largest density at which the merge still wins is what
  * a sample is read sparse below. */

 const std::size_t m = 4096;
 const std::size_t n = 256;    // 8 MB of samples, i.e., past the cache
 const std::size_t pairs = 64;

 std::mt19937 rng( 1 );
 std::uniform_real_distribution< double > unif( -1 , 1 );
 std::uniform_real_distribution< double > coin( 0 , 1 );
 std::uniform_int_distribution< std::size_t > which( 0 , n - 1 );

 static const double densities[] = { 0.02 , 0.05 , 0.1 , 0.15 , 0.2 , 0.3 ,
                                     0.4 , 0.5 , 0.6 , 0.8 };

 double crossing = 0;
 double dot = 0 , merge = 0;

 for( auto dns : densities ) {
  // the data set, dense and as the lists of the nonzeroes of each sample,
  // which is the pair of representations the kernel reads
  std::vector< double > X( n * m , 0 );
  std::vector< std::size_t > Xp( n + 1 , 0 ) , Xi;
  std::vector< double > Xv;

  for( std::size_t i = 0 ; i < n ; ++i ) {
   for( std::size_t j = 0 ; j < m ; ++j )
    if( coin( rng ) < dns ) {
     const double v = unif( rng );
     X[ i * m + j ] = v;
     Xi.push_back( j );
     Xv.push_back( v );
     }
   Xp[ i + 1 ] = Xi.size();
   }

  // the pairs are drawn once and read by both, so that the two walk the
  // same samples in the same order
  std::vector< std::size_t > pa( pairs ) , pb( pairs );
  for( std::size_t t = 0 ; t < pairs ; ++t ) {
   pa[ t ] = which( rng );
   pb[ t ] = which( rng );
   }

  const double one_dot = timed( [ & ]() {
   double acc = 0;
   for( std::size_t t = 0 ; t < pairs ; ++t ) {
    const double * x = X.data() + pa[ t ] * m;
    const double * z = X.data() + pb[ t ] * m;
    double d = 0;
    for( std::size_t j = 0 ; j < m ; ++j )
     d += x[ j ] * z[ j ];
    acc += d;
    }
   return( acc );
   } , sink ) / double( pairs );

  const double one_merge = timed( [ & ]() {
   double acc = 0;
   for( std::size_t t = 0 ; t < pairs ; ++t ) {
    std::size_t a = Xp[ pa[ t ] ] , b = Xp[ pb[ t ] ];
    const std::size_t ea = Xp[ pa[ t ] + 1 ] , eb = Xp[ pb[ t ] + 1 ];
    double d = 0;
    while( ( a < ea ) && ( b < eb ) )
     if( Xi[ a ] == Xi[ b ] ) {
      d += Xv[ a ] * Xv[ b ];
      ++a;
      ++b;
      }
     else
      if( Xi[ a ] < Xi[ b ] )
       ++a;
      else
       ++b;
    acc += d;
    }
   return( acc );
   } , sink ) / double( pairs );

  if( ! dot )        // the dense one does not depend on the density, and is
   dot = one_dot;    // read off the first data set

  if( one_merge < one_dot ) {
   crossing = dns;
   merge = one_merge / ( 2 * dns * double( m ) );
   }
  else
   break;
  }

 // what starting a thread and waiting for it costs, which is what says how
 // much work is worth handing over to one
 const double thread = timed( [ & ]() {
  double d = 0;
  std::thread t( [ & d ]() { d = 1; } );
  t.join();
  return( d );
  } , sink );

 const double memory = total_memory();
 const unsigned cores = std::max< unsigned >(
                                  1 , std::thread::hardware_concurrency() );

 // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

 const std::string name = path + "/SVMBlockArch.h";

 std::ofstream out( name );
 if( ! out.is_open() ) {
  std::cerr << "svm_arch: cannot write " << name << std::endl;
  return( 1 );
  }

 out << std::scientific << std::setprecision( 6 )
     << "/*-------------------------------------------------------------"
     << "-------------*/\n"
     << "/*-------------------------- File SVMBlockArch.h ---------------"
     << "-------------*/\n"
     << "/*-------------------------------------------------------------"
     << "-------------*/\n"
     << "/** @file\n"
     << " *\n"
     << " * The properties of the machine that the defaults of SVMBlock "
     << "depend on,\n"
     << " * measured by svm_arch at the first compilation: this file is the "
     << "machine's\n"
     << " * own and is not part of the sources [see svm_arch.cpp].\n"
     << " */\n"
     << "/*-------------------------------------------------------------"
     << "-------------*/\n\n"
     << "#ifndef __SVMBlockArch\n"
     << " #define __SVMBlockArch\n\n"
     << "/// bytes of memory the machine has\n"
     << "#define SVMBlock_ARCH_MEMORY " << memory << "\n\n"
     << "/// cores the machine has\n"
     << "#define SVMBlock_ARCH_CORES " << cores << "\n\n"
     << "/// seconds one entry of a dense inner product takes\n"
     << "#define SVMBlock_ARCH_DOT " << dot / double( m ) << "\n\n"
     << "/// seconds one entry of the merge of two lists of nonzeroes takes\n"
     << "#define SVMBlock_ARCH_MERGE " << merge << "\n\n"
     << "/// the density below which the merge of the two lists still wins\n"
     << "#define SVMBlock_ARCH_DENSITY " << crossing << "\n\n"
     << "/// seconds starting a thread and waiting for it takes\n"
     << "#define SVMBlock_ARCH_THREAD " << thread << "\n\n"
     << "#endif  /* SVMBlockArch.h included */\n\n"
     << "/*-------------------------------------------------------------"
     << "-------------*/\n"
     << "/*---------------------- End File SVMBlockArch.h ---------------"
     << "-------------*/\n"
     << "/*-------------------------------------------------------------"
     << "-------------*/\n";

 out.close();

 std::cout << "svm_arch: " << memory / ( 1024 * 1024 * 1024.0 )
           << " GB, " << cores << " cores, dot " << dot / double( m )
           << " s, merge " << merge << " s, crossing density " << crossing
           << ", thread " << thread << " s" << std::endl;

 return( 0 );

 }  // end( main )

/*--------------------------------------------------------------------------*/
/*------------------------ End File svm_arch.cpp ---------------------------*/
/*--------------------------------------------------------------------------*/
