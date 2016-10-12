#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include <boost/iterator/counting_iterator.hpp>

#include <mkl_cblas.h>

// use std::for_each(std::execution::par, ... ) in C++17 instead
template <typename UnaryFunction>
void omp_parallel_for(int first, int last, UnaryFunction f) {
#pragma omp parallel for // OpenMP 2.0 compatibility:  signed loop variable
  for (int i = first; i < last; ++i) {
    f(i);
  }
}

template <typename ForwardIt> void fill_random(ForwardIt begin, ForwardIt end) {
  std::random_device rndDev;
  std::mt19937 rndEng{rndDev()};
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  std::uniform_real_distribution<T> dist{-1.0, 1.0};

  std::generate(begin, end, [&] { return dist(rndEng); });
}

enum class Mode {
  SerialIkj, //
  SerialIjk, //
  Parallel,  //
  Blas       //
};

template <typename T>
void SerialIkj(const T *a, const T *b, T *c, std::size_t aRows,
               std::size_t aCols, std::size_t bCols) {
  for (std::size_t i{0}; i < aRows; ++i) {
    for (std::size_t k{0}; k < aCols; ++k) {
      for (std::size_t j{0}; j < bCols; ++j) {
        c[i * bCols + j] += a[i * aCols + k] * b[k * bCols + j];
      }
    }
  }
}

template <typename T>
void SerialIjk(const T *a, const T *b, T *c, std::size_t aRows,
               std::size_t aCols, std::size_t bCols) {
  for (std::size_t i{0}; i < aRows; ++i) {
    for (std::size_t j{0}; j < bCols; ++j) {
      for (std::size_t k{0}; k < aCols; ++k) {
        c[i * bCols + j] += a[i * aCols + k] * b[k * bCols + j];
      }
    }
  }
}

template <typename T>
void parallelGemm(const T *a, const T *b, T *c, std::size_t aRows,
                  std::size_t aCols, std::size_t bCols) {
  omp_parallel_for(0, aRows, [&](auto i) {
    for (std::size_t k{0}; k < aCols; ++k) {
      for (std::size_t j{0}; j < bCols; ++j) {
        c[i * bCols + j] += a[i * aCols + k] * b[k * bCols + j];
      }
    }
  });
}

void blasGemm(const float *a, const float *b, float *c, std::size_t aRows,
              std::size_t aCols, std::size_t bCols) {
  const auto m = aRows;
  const auto k = aCols;
  const auto n = bCols;
  const float alf = 1;
  const float bet = 0;

  cblas_sgemm(CblasRowMajor, // CBLAS_LAYOUT layout,
              CblasNoTrans,  // CBLAS_TRANSPOSE TransA,
              CblasNoTrans,  // CBLAS_TRANSPOSE TransB,
              m,             // const int M,
              n,             // const int N,
              k,             // const int K,
              alf,           // const float alpha,
              a,             // const float *A,
              k,             // const int lda,
              b,             // const float *B,
              n,             // const int ldb,
              bet,           // const float beta,
              c,             // float *C,
              n              // const int ldc
              );
}

void blasGemm(const double *a, const double *b, double *c, std::size_t aRows,
              std::size_t aCols, std::size_t bCols) {
  const auto m = aRows;
  const auto k = aCols;
  const auto n = bCols;
  const double alf = 1;
  const double bet = 0;

  cblas_dgemm(CblasRowMajor, // CBLAS_LAYOUT layout,
              CblasNoTrans,  // CBLAS_TRANSPOSE TransA,
              CblasNoTrans,  // CBLAS_TRANSPOSE TransB,
              m,             // const int M,
              n,             // const int N,
              k,             // const int K,
              alf,           // const float alpha,
              a,             // const float *A,
              k,             // const int lda,
              b,             // const float *B,
              n,             // const int ldb,
              bet,           // const float beta,
              c,             // float *C,
              n              // const int ldc
              );
}

template <typename T> auto timed(T &&f) {
  const auto starttime = std::chrono::high_resolution_clock::now();
  f();
  return std::chrono::duration<double>{
      std::chrono::high_resolution_clock::now() - starttime}
      .count();
}

// dispatching to the impls
template <typename T>
void gemm(const Mode mode, const T *a, const T *b, T *c, std::size_t aRows,
          std::size_t aCols, std::size_t bCols) {
  switch (mode) {
  case Mode::SerialIkj:
    SerialIkj(a, b, c, aRows, aCols, bCols);
    break;
  case Mode::SerialIjk:
    SerialIjk(a, b, c, aRows, aCols, bCols);
    break;
  case Mode::Parallel:
    parallelGemm(a, b, c, aRows, aCols, bCols);
    break;
  case Mode::Blas:
    blasGemm(a, b, c, aRows, aCols, bCols);
    break;
  default:
    std::cerr << "unsupported mode!\n";
    std::exit(EXIT_FAILURE);
  }
}

[[noreturn]] void failInvalidArgs() {
  std::cout << "invalid args!\nusage: prog "
               "serialikj|serialijk|parallel|optimized|blas\n";
  std::exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  using Real = float;
  if (argc != 2) {
    failInvalidArgs();
  }
  const auto mode = [&]() {
    auto const m{std::string(argv[1])};
    if (m == "serialikj") {
      return Mode::SerialIkj;
    } else if (m == "serialijk") {
      return Mode::SerialIjk;
    } else if (m == "parallel") {
      return Mode::Parallel;
    } else if (m == "blas") {
      return Mode::Blas;
    } else {
      failInvalidArgs();
    }
  }();

  const auto minSize = 2u;
  const auto maxSize = 2048;
  const auto maxRepetitions = 2048u;

  // do measurements with increasing matrix dimensions and decreasing
  // repetitions to keep wall clock time short
  auto repetitions = maxRepetitions;
  for (std::size_t size{minSize}; size <= maxSize;
       size *= 2, repetitions /= 2) {

    /// set up data
    const auto aRows = size;
    const auto aCols = size;
    const auto bRows = size;
    const auto bCols = size;

    std::vector<Real> a(aRows * aCols); // m * k
    std::vector<Real> b(bRows * bCols); // k * n
    std::vector<Real> c(aRows * bCols); // m * n

    fill_random(std::begin(a), std::end(a));
    fill_random(std::begin(b), std::end(b));

    /// warm up the caches
    gemm(mode, a.data(), b.data(), c.data(), aRows, aCols, bCols);

    /// timed computations
    auto time = 0.0;
    for (std::size_t r{0}; r < repetitions; ++r) {
      std::fill(std::begin(c), std::end(c), 0);
      time += timed([&]() {
        gemm(mode, a.data(), b.data(), c.data(), aRows, aCols, bCols);
      });
      // access the output to prevent unwanted compiler optimizations
      std::ostream null{nullptr};
      std::copy(std::begin(c), std::end(c), std::ostream_iterator<Real>{null});
    }
    time /= repetitions; // get avg time per call

    std::cout << size << ";" << time << '\n';
  }
}