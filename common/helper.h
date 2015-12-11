
// =================================================================================================
// This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
// CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
//
// Author(s):
//   Valeriu Codreanu <valeriu.codreanu@surfsara.nl>
//
// This file contains helper functions for the Dense Linear Algebra examples.
// =================================================================================================

#include <assert.h>
#include <math.h>
#ifdef __OPENMP
#include <omp.h>
#endif

#include <sys/time.h>
#include <ctime>

/* Remove if already defined */
typedef long long int64; typedef unsigned long long uint64;

/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
 * windows and linux. */

uint64 get_time_uint64()
{
 /* Linux */
 struct timeval tv;

 gettimeofday(&tv, NULL);

 uint64 ret = tv.tv_usec;
 /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
 ret /= 1000;

 /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
 ret += (tv.tv_sec * 1000);

 return ret;
}



inline bool
compareReference(const float *reference, const float *data,
               const unsigned int len, const float epsilon)
{
    assert(epsilon >= 0);

    float error = 0;
    float ref = 0;
#ifdef __OPENMP
#pragma omp parallel for
#endif
    for (unsigned int i = 0; i < len; ++i)
    {

        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float norm_ref = sqrtf(ref);

    if (fabs(ref) < 1e-7)
    {
#ifdef _DEBUG
        std::cerr << "ERROR, reference l2-norm is 0\n";
#endif
        return false;
    }

    float norm_error = sqrtf(error);
    error = norm_error / norm_ref;
    bool result = error < epsilon;
#ifdef _DEBUG

    if (! result)
    {
        std::cerr << "ERROR, l2-norm error "
                  << error << " is greater than epsilon " << epsilon << "\n";
    }

#endif

    return result;
}


