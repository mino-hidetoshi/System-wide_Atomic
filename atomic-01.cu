/*
    Compilation: 
    
    nvcc -arch=sm_60 atomic-01.cu
    
    Execution example:
    
    $ CUDA_VISIBLE_DEVICES=0 ./a.out 
    atomic-01.cu 
    5000000 
    
    $ CUDA_VISIBLE_DEVICES=0,1 ./a.out 
    atomic-01.cu 
    5000269
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
using namespace std;

#define CU_SAFE(stmt) \
  do { \
    cudaError_t status = stmt; \
    if (status != cudaSuccess) { \
       printf("ERROR: %s in %s:%d.\n", cudaGetErrorString(status), \
       __FILE__, __LINE__); \
       exit( 1 );  \
    } \
  } while (0)

#define WS 32   /* Warp size */
#define AllLanes (0xffffffff)

typedef long long unsigned u64;

/* = Managed ( Unified ) =================== */

__managed__ int itask_m = 0;

__managed__ u64 total_count_m = 0ULL;

/* ========================================= */

__global__  void Kernel( u64* v_d, int ntask, int nv );

void Count( )
{
  int ntask = 5000000;
  int nv = 50 * WS;

  // Multi Device Handling ----------------------

  int ndev;
  CU_SAFE( cudaGetDeviceCount( &ndev ) );

  u64* v_d[ ndev ]; 

  for( int idev = 0 ; idev < ndev; idev ++ ) {

    CU_SAFE( cudaSetDevice( idev ) );
    cudaDeviceProp prop;
    CU_SAFE( cudaGetDeviceProperties( &prop, idev ) );

    int nblocks = prop.multiProcessorCount;

    CU_SAFE( cudaMalloc( v_d + idev, sizeof(u64) * nv *( 1 + nblocks ) ) );
    /* contents of this array don't matter. */

    Kernel <<< nblocks, WS >>> ( v_d[ idev ], ntask, nv );
  }

  for( int idev = 0 ; idev < ndev; idev ++ ) {

    CU_SAFE( cudaSetDevice( idev ) );
    CU_SAFE( cudaFree( v_d[ idev ] ) );  
  }

  return; 
}

__device__ int vcopy( u64* vp, int nv, u64* vpn ) 
{
   assert( ( nv % WS ) == 0 );
   int j = 0;
   for( int i = 0; i < nv; i += WS ) {
      vpn[ j + threadIdx.x ] = vp[ i + threadIdx. x ];
      j += WS;
   }
   return j;
}

__global__ void Kernel( u64* v_d, int ntask, int nv )
{
  u64* vb = v_d + nv * ( blockIdx.x + 1 );

  u64 count = 0ULL;

  for(   ; /* i_task < ntask */; /* i_task ++ */ ) {

     int itask;
     if( threadIdx.x == 0 ) {
        itask = atomicAdd_system( &itask_m, 1 );
     }
     itask = __shfl_sync( AllLanes, itask, 0 ); 

     if( itask >= ntask ) break;

     vcopy( v_d, nv, vb ); // just wasting time.
     vcopy( v_d, nv, vb );
     vcopy( v_d, nv, vb );
     vcopy( v_d, nv, vb );
     vcopy( v_d, nv, vb );
     vcopy( v_d, nv, vb );
     vcopy( v_d, nv, vb );
     vcopy( v_d, nv, vb );
     vcopy( v_d, nv, vb );
     vcopy( v_d, nv, vb );

     count ++;
  }

  if( threadIdx.x == 0 ) 
  {
     atomicAdd_system( &total_count_m, count );
  }
}

int main( int argc, char* argv[] )
{
   fprintf( stderr, "%s\n", __FILE__ );

   total_count_m = 0ULL; // initialize managed variable
   itask_m = 0; 

   Count( );

   printf( " %15llu \n", total_count_m ); 

   return 0;
}

