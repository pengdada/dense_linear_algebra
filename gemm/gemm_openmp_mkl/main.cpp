#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>

#include <mkl_cblas.h>

/// helper functions
template <typename ForwardIt> void fill_random(ForwardIt begin, ForwardIt end) {
	std::random_device rndDev;
	std::mt19937 rndEng{rndDev()};
	using T = typename std::iterator_traits<ForwardIt>::value_type;
	std::uniform_real_distribution<T> dist{-1.0, 1.0};

	std::generate(begin, end, [&] { return dist(rndEng); });
}

template <typename T> auto timed(T&& f) {
	const auto starttime = std::chrono::high_resolution_clock::now();
	f();
	return std::chrono::duration<double>{
	    std::chrono::high_resolution_clock::now() - starttime}
	    .count();
}

template <typename Map>[[noreturn]] void failInvalidArgs(const Map& modes) {
	std::cerr << "invalid args!\nusage: prog "
	          << boost::algorithm::join(
	                 modes | boost::adaptors::transformed(
	                             [](const auto& p) { return p.first; }),
	                 "|")
	          << "\n";
	std::exit(EXIT_FAILURE);
}

/// gemm implementations
template <typename T>
void serialIkj(const T* a, const T* b, T* c, std::size_t aRows,
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
void serialIjk(const T* a, const T* b, T* c, std::size_t aRows,
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
void parallelGemm(const T* a, const T* b, T* c, std::size_t aRows,
                  std::size_t aCols, std::size_t bCols) {
// use std::for_each(std::execution::par, ... ) in
// C++17 instead
#pragma omp parallel for
	for (std::size_t i{0}; i < aRows; ++i) {
		for (std::size_t k{0}; k < aCols; ++k) {
			for (std::size_t j{0}; j < bCols; ++j) {
				c[i * bCols + j] += a[i * aCols + k] * b[k * bCols + j];
			}
		}
	}
}

// float overload
void blasGemm(const float* a, const float* b, float* c, std::size_t aRows,
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

// double overload
void blasGemm(const double* a, const double* b, double* c, std::size_t aRows,
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
	            alf,           // const double alpha,
	            a,             // const double *A,
	            k,             // const int lda,
	            b,             // const double *B,
	            n,             // const int ldb,
	            bet,           // const double beta,
	            c,             // double *C,
	            n              // const int ldc
	            );
}

int main(int argc, char* argv[]) {
	using Real = float;
	using gemmFuncType = void(const Real*, const Real*, Real*, std::size_t,
	                          std::size_t, std::size_t);
	using gemmFuncPrtType = void (*)(const Real*, const Real*, Real*,
	                                 std::size_t, std::size_t, std::size_t);

	// this map manages all implementations
	std::map<std::string, std::function<gemmFuncType>> modes{
	    {"serialikj", &serialIkj<Real>},
	    {"serialijk", &serialIjk<Real>},
	    {"parallel", &parallelGemm<Real>},
	    {"blas", static_cast<gemmFuncPrtType>(&blasGemm)}};

	// validate cmd args and select gemm implementation
	if (argc != 2) { failInvalidArgs(modes); }
	const auto modeArg{std::string(argv[1])};
	if (modes.count(modeArg) != 1) { failInvalidArgs(modes); }
	const auto gemm = modes[modeArg];

	// benchmark sizes
	const auto minSize = 2u;
	const auto maxSize = 2048u;
	const auto maxRepetitions = 2048u;

	// do measurements with increasing matrix dimensions and decreasing
	// repetitions to keep wall clock time short
	std::size_t repetitions{};
	std::size_t size{};
	for (size = minSize, repetitions = maxRepetitions; size <= maxSize;
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

		// create reference solution matrix using serialIkj
		std::vector<Real> reference(c.size());
		serialIkj(a.data(), b.data(), reference.data(), aRows, aCols, bCols);

		/// warm up the caches
		gemm(a.data(), b.data(), c.data(), aRows, aCols, bCols);

		/// timed computations
		auto time = 0.0;
		for (std::size_t r{0}; r < repetitions; ++r) {
			std::fill(std::begin(c), std::end(c), 0);
			time += timed([&]() {
				gemm(a.data(), b.data(), c.data(), aRows, aCols, bCols);
			});

			// validate the result c
			if (!std::equal(std::begin(c), std::end(c), std::begin(reference),
			                std::end(reference),
			                [](const Real& l, const Real& r) {
				                return std::abs(l - r) < 1.0e-3;
				            })) {
				std::cerr << "validation error!\n";
				return EXIT_FAILURE;
			}
		}
		time /= repetitions; // get avg time per call

		std::cout << size << ";" << time << '\n';
	}
}