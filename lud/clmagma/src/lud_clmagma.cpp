// =================================================================================================
// This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
// CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
//
// Author(s):
//    Mariusz Uchronski <mariusz.uchronski@pwr.edu.pl>
//
// =================================================================================================

#include <stdio.h>
#include "magma.h"

int main(int argc, char *argv[])
{
  float          *h_A;
  magmaFloat_ptr d_A;
  magma_int_t     *ipiv;
  magma_int_t     M, N, n2, lda, ldda, info, min_mn;
  magma_int_t     status   = 0;

  /* Initialize */
  magma_queue_t  queue;
  magma_device_t devices[MagmaMaxGPUs];
  magma_int_t num = 0;
  magma_int_t err;

  magma_init();
  err = magma_getdevices( devices, MagmaMaxGPUs, &num );
  if ( err != 0 or num < 1 ) {
    fprintf( stderr, "magma_getdevices failed: %d\n", (int) err );
    exit(-1);
  }
  err = magma_queue_create( devices[0], &queue );
  if ( err != 0 ) {
    fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
    exit(-1);
  }

  lda    = M;
  n2     = lda*N;
  ldda   = ((M+31)/32)*32;

  //TESTING_MALLOC_DEV( d_A, float, ldda*N );
  magma_malloc( &d_A, (ldda*N)*sizeof(float) );
  //TESTING_MALLOC_CPU( h_A, float, n2     );
  magma_malloc_cpu( (void**)&h_A, (n2)*sizeof(float) );
  magma_ssetmatrix( M, N, h_A, lda, d_A, 0, ldda, queue );
  magma_sgetrf_gpu( M, N, d_A, 0, ldda, ipiv, queue, &info );

  magma_free_cpu(h_A);
  magma_free(d_A);
  magma_finalize();
  return 0;
}
