/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Intel Corporation.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "./Config.h"
#include "./FbgemmBuild.h"
#include "./UtilsAvx2.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <type_traits>

#if FBGEMM_PARALLEL_OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

namespace fbgemm {

/**
 * @brief Helper struct to type specialize for uint8 and int8 together.
 */
template <typename T>
struct is_8bit {
  static constexpr bool value =
      std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
};

/**
 * @brief Typed enum to specify matrix operations.
 */
enum class matrix_op_t { NoTranspose, Transpose };

/**
 * @brief Typed enum for supported instruction sets.
 */
enum class inst_set_t {
  anyarch,
  avx2,
  avx512,
  avx512_ymm,
  avx512_vnni,
  avx512_vnni_ymm
};

/**
 * @brief Typed enum for optimized paths for convolutions
 */
enum class optimized_conv_t {
  depthwise,
  groupwise,
  pointwise,
  fastpath1d,
  im2col,
  directconv
};

/**
 * @brief Typed enum for implementation type.
 *
 * ref is reference and opt is optimized.
 */
enum class impl_type_t { ref, opt };

/**
 * @brief Typed enum to specify data layout.
 * KCX can be KCRS format or KCTRS format (e.g., for 3-D convolutions)
 * KXC can be KRSC format or KTRSC format (e.g., for 3-D convolutions)
 */
enum class FBGEMM_ENUM_CLASS_API layout_t { KCX, KXC };

/**
 * @brief A function to compare data in two buffers for closeness/equality.
 */
template <typename T>
FBGEMM_API int compare_buffers(
    const T* ref,
    const T* test,
    int m,
    int n,
    int ld,
    size_t max_mismatches_to_report,
    float atol = 1e-3);

/**
 * @brief Debugging helper.
 */
template <typename T>
void printMatrix(
    matrix_op_t trans,
    const T* inp,
    size_t R,
    size_t C,
    size_t ld,
    std::string name);

/**
 * @brief Transpose a matrix.
 *
 * @param M the number of rows of input matrix
 * @param N the number of columns of input matrix
 */
template <typename T>
FBGEMM_API void transpose_simd(
    int64_t M,
    int64_t N,
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst);

/**
 * @brief Explicitly set instruction set to be used
 */
FBGEMM_API void fbgemmForceIsa(inst_set_t);

/**
 * @brief Enable AVX512-256 path for Intel(r) Xeon(r) D servers
 */
FBGEMM_API void fbgemmEnableAvx512Ymm(bool);

/**
 * @brief Are we running on a Xeon-D cpu?
 */
FBGEMM_API bool fbgemmIsIntelXeonD();

/**
 * @brief Are we running on a AVX512 supported cpu?
 */
FBGEMM_API bool fbgemmHasAvx512Support();

/**
 * @brief Are we running on a AVX2 supported cpu?
 */
FBGEMM_API bool fbgemmHasAvx2Support();

/**
 * @brief Are we running on a AVX512_VNNI supported cpu?
 */
FBGEMM_API bool fbgemmHasAvx512VnniSupport();

/**
 * @brief Are we running on a ARM Neon supported cpu?
 */
FBGEMM_API bool fbgemmHasArmNeonSupport();

/**
 * @brief Retrieve current CPU instruction set
 */
FBGEMM_API inst_set_t fbgemmInstructionSet();

/**
 * @brief Is ISA is wide vector ZMM
 */
FBGEMM_API bool isZmm(inst_set_t);

/**
 * @brief Is ISA is wide vector ZMM
 */
FBGEMM_API bool isYmm(inst_set_t);

/**
 * @brief Helper struct to enable autotuning of FBGEMM packing and kernels.
 *
 * This structure is optional. If not used, the default values for these
 * parameters are picked up from PackingTraits-inl.h. Please see this
 * file for details on these parameters.
 */
struct FBGEMM_API BlockingFactors {
  int MR;
  int NR;
  int NR_MIN;
  int ROW_INTERLEAVE;
  int MCB;
  int KCB;
  int NCB;
};

/**
 * @brief A struct to represent the partition information for the threads on the
 * m and n dimensions.
 */
struct FBGEMM_API thread_type_t {
  int g_num_threads;
  int m_num_threads;
  int n_num_threads;
  int g_thread_id;
  int m_thread_id;
  int n_thread_id;

  std::string toString() const {
    std::string out = "";
    out += "g num threads: " + std::to_string(g_num_threads) + ", ";
    out += "m num threads: " + std::to_string(m_num_threads) + ", ";
    out += "n num threads: " + std::to_string(n_num_threads) + ", ";
    out += "g thread id: " + std::to_string(g_thread_id) + ", ";
    out += "m thread id: " + std::to_string(m_thread_id) + ", ";
    out += "n thread id: " + std::to_string(n_thread_id);
    return out;
  }
};

/**
 * @brief A heuristic algorithm to partition the threads across m and n
 * dimensions for parallelization, ensuring the ratio between the number of rows
 * allocated to each thread in the m dimension and the number of columns
 * allocated to each thread in the n dimension is approximately aspect_ratio.
 *
 * The less aspect_ratio is, the more favorable it is to parallelize the m
 * dimension over the n dimension.
 */
FBGEMM_API int fbgemmGet2DPartition(
    int m,
    int n,
    int nthreads,
    int n_align,
    double aspect_ratio);

/**
 * @brief A heuristic way to partition the threads across g, m and n dimensions
 * for parallelization.
 */
FBGEMM_API thread_type_t fbgemmGetThreadPartition(
    int g,
    int m,
    int n,
    int num_threads,
    int thread_id,
    int n_align = 64);

template <int SIZE, typename T = std::int32_t>
std::string arrayToString(const std::array<T, SIZE>& inp) {
  std::string out = "[";
  for (int i = 0; i < SIZE; ++i) {
    out += std::to_string(inp[i]);
    out += (i != SIZE - 1) ? std::string(", ") : std::string("]");
  }
  return out;
}

template <typename accT = std::int32_t>
bool isValidBlockingFactor(BlockingFactors* param) {
  constexpr bool is_32bit = std::is_same<accT, int32_t>::value;
  constexpr bool is_16bit = std::is_same<accT, int16_t>::value;
  static const auto iset = fbgemmInstructionSet();

  if (is_32bit) {
    if (param->ROW_INTERLEAVE != 4)
      return false;

    if (isZmm(iset)) {
      if (param->NR_MIN != 16 || param->NR % param->NR_MIN)
        return false;
    } else if (isYmm(iset)) {
      if (param->NR_MIN != 8 || param->NR % param->NR_MIN)
        return false;
    }
  } else if (is_16bit) {
    if (param->ROW_INTERLEAVE != 2)
      return false;

    if (isZmm(iset)) {
      if (param->NR_MIN != 32 || param->NR % param->NR_MIN)
        return false;
    } else if (isYmm(iset)) {
      if (param->NR_MIN != 16 || param->NR % param->NR_MIN)
        return false;
    }
  }

  if (param->MCB % param->MR)
    return false;
  if (param->NCB % param->NR)
    return false;
  if (isZmm(iset)) {
    if (is_32bit) {
      // Zmm register usage for C
      if (param->MR * (param->NR / param->NR_MIN) > 28)
        return false;
    } else if (is_16bit) {
      // Zmm register usage for C + one row for loading B
      if ((param->MR * (param->NR / param->NR_MIN) +
           (param->NR / param->NR_MIN)) > 28)
        return false;
    }

  } else if (isYmm(iset)) {
    if (param->MR * (param->NR / param->NR_MIN) > 12)
      return false;
  }
  return true;
}

/**
 * @brief Partition work across given number of threads
 *
 * @param start Given thread_id should execute starting from the index
 *              start
 * @param stop Given thread_id should stop executing at the index stop
 *
 * i.e., the loop should be equivalent to for(int i = start; i < end; ++i)
 */
FBGEMM_API void fbgemmPartition1D(
    int thread_id,
    int num_threads,
    std::int64_t total_work,
    std::int64_t& start,
    std::int64_t& end);

/**
 * @brief Partition work across given number of threads in blocks
 *        of size block_size. Each thread gets a multiple of block_size
 *        work or nothing, except the last one. The last one might
 *        receive the fringe case.
 *
 * @param start Given thread_id should execute starting from the index
 *              start
 * @param stop Given thread_id should stop executing at the index stop
 *
 * The loop can be equivalent to for(int i = start; i < end; i+=block_size)
 * except for the last thread. (i.e., thread_id = num_threads - 1)
 *
 * Example 1: block_size = 2, num_threads = 2
 *  total_work  start(th 0) end(th 0) start(th 1) end(th 1)
 *      4         0           2          2          4
 *      5         0           2          2          5
 *
 * Example 2: block_size = 2, num_threads = 3
 *  total_work  start(th 0) end(th 0) start(th 1) end(th 1)
 *      4         0           2          2          4
 *      5         0           2          2          4
 *
 *  total_work  start(th 2) end(th 2)
 *      4         4           4
 *      5         4           5
 *
 * Example 3: block_size = 2, num_threads = 4
 *  total_work  start(th 0) end(th 0) start(th 1) end(th 1)
 *      4         0           2          2          4
 *      5         0           2          2          4
 *
 *  total_work  start(th 2) end(th 2) start(th 3) end(th 3)
 *      4         4           4          4          4
 *      5         4           4          4          5
 */
FBGEMM_API void fbgemmPartition1DBlocked(
    int thread_id,
    int num_threads,
    std::int64_t total_work,
    int block_size,
    std::int64_t& start,
    std::int64_t& end);

namespace {

// implementation taken from pytorch/c10/util/llvmMathExtras.h
template <typename T>
size_t count_leading_zeros(T val) {
    if (!val)
      return std::numeric_limits<T>::digits;

    size_t zero_bits = 0;
    for (T shift = std::numeric_limits<T>::digits >> 1; shift; shift >>= 1) {
      T tmp = val >> shift;
      if (tmp)
        val = tmp;
      else
        zero_bits |= shift;
    }
    return zero_bits;
}

// histogram size per thread
constexpr int RDX_HIST_SIZE = 256;

#define COMBINE_PREFIX_SUM_IN_RANGE(sum, prev_sum, bins_beg, bins_end, hist, hist_ps, nthreads) \
  for (int bins = bins_beg; bins < bins_end; ++bins) { \
    for (int t = 0; t < nthreads; ++t) { \
      sum += histogram[t * RDX_HIST_SIZE + bins]; \
      histogram_ps[t * RDX_HIST_SIZE + bins] = prev_sum; \
      prev_sum = sum; \
    } \
  }

void combine_prefix_sum(
    int nthreads,
    int elements_count,
    int* histogram,
    int* histogram_ps) {
  int sum = 0, prev_sum = 0;
  COMBINE_PREFIX_SUM_IN_RANGE(sum, prev_sum, 0, RDX_HIST_SIZE, histogram, histogram_ps, nthreads);
  histogram_ps[RDX_HIST_SIZE * nthreads] = prev_sum;
  // TODO(dszwicht): Is assert sufficient? In most cases, it will work only in
  // debug build.
  assert(prev_sum == elements_count);
  // Suppress unused variable warning
  (void)elements_count;
}

void combine_prefix_sum_for_msb(
    int nthreads,
    int elements_count,
    int* histogram,
    int* histogram_ps) {
  int sum = 0, prev_sum = 0;
  COMBINE_PREFIX_SUM_IN_RANGE(sum, prev_sum, 128, RDX_HIST_SIZE, histogram, histogram_ps, nthreads);
  COMBINE_PREFIX_SUM_IN_RANGE(sum, prev_sum, 0, 128, histogram, histogram_ps, nthreads);
  histogram_ps[RDX_HIST_SIZE * (nthreads - 1) + 127] = prev_sum;
  // TODO(dszwicht): Is assert sufficient? In most cases, it will work only in
  // debug build.
  assert(prev_sum == elements_count);
  // Suppress unused variable warning
  (void)elements_count;
}

template <typename K, typename V>
void radix_sort_kernel(
    K* input_keys,
    V* input_values,
    K* output_keys,
    V* output_values,
    int elements_count,
    int* histogram,
    int* histogram_ps,
    int pass,
    bool pass_with_sign_bit = false) {
  const int tid = omp_get_thread_num();
  const int nthreads = omp_get_num_threads();
  const int elements_count_4 = elements_count / 4 * 4;

  int* local_histogram = &histogram[RDX_HIST_SIZE * tid];
  int* local_histogram_ps = &histogram_ps[RDX_HIST_SIZE * tid];

  // Step 1: compute histogram
  for (int i = 0; i < RDX_HIST_SIZE; i++) {
    local_histogram[i] = 0;
  }

#pragma omp for schedule(static)
  for (int64_t i = 0; i < elements_count_4; i += 4) {
    K key_1 = input_keys[i];
    K key_2 = input_keys[i + 1];
    K key_3 = input_keys[i + 2];
    K key_4 = input_keys[i + 3];

    local_histogram[(key_1 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_2 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_3 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_4 >> (pass * 8)) & 0xFF]++;
  }
  if (tid == (nthreads - 1)) {
    for (int64_t i = elements_count_4; i < elements_count; ++i) {
      K key = input_keys[i];
      local_histogram[(key >> (pass * 8)) & 0xFF]++;
    }
  }
#pragma omp barrier
  // Step 2: prefix sum
  if (tid == 0) {
    if (pass_with_sign_bit) {
      combine_prefix_sum_for_msb(nthreads, elements_count, histogram, histogram_ps);
    } else {
      combine_prefix_sum(nthreads, elements_count, histogram, histogram_ps);
    }
  }
#pragma omp barrier

  // Step 3: scatter
#pragma omp for schedule(static)
  for (int64_t i = 0; i < elements_count_4; i += 4) {
    K key_1 = input_keys[i];
    K key_2 = input_keys[i + 1];
    K key_3 = input_keys[i + 2];
    K key_4 = input_keys[i + 3];

    const int bin_1 = (key_1 >> (pass * 8)) & 0xFF;
    const int bin_2 = (key_2 >> (pass * 8)) & 0xFF;
    const int bin_3 = (key_3 >> (pass * 8)) & 0xFF;
    const int bin_4 = (key_4 >> (pass * 8)) & 0xFF;

    int pos;
    pos = local_histogram_ps[bin_1]++;
    output_keys[pos] = key_1;
    output_values[pos] = input_values[i];
    pos = local_histogram_ps[bin_2]++;
    output_keys[pos] = key_2;
    output_values[pos] = input_values[i + 1];
    pos = local_histogram_ps[bin_3]++;
    output_keys[pos] = key_3;
    output_values[pos] = input_values[i + 2];
    pos = local_histogram_ps[bin_4]++;
    output_keys[pos] = key_4;
    output_values[pos] = input_values[i + 3];
  }
  if (tid == (nthreads - 1)) {
    for (int64_t i = elements_count_4; i < elements_count; ++i) {
      K key = input_keys[i];
      const int pos = local_histogram_ps[(key >> (pass * 8)) & 0xFF]++;
      output_keys[pos] = key;
      output_values[pos] = input_values[i];
    }
  }
}

} // namespace

// Inline implementation as we cannot predict all template instantiation.
template <typename K, typename V>
std::pair<K*, V*> radix_sort_parallel(
    K* inp_key_buf,
    V* inp_value_buf,
    K* tmp_key_buf,
    V* tmp_value_buf,
    int64_t elements_count,
    int64_t max_value,
    bool maybe_with_neg_vals = false) {
  const int maxthreads = omp_get_max_threads();
  alignas(64) int histogram[RDX_HIST_SIZE * maxthreads];
  alignas(64) int histogram_ps[RDX_HIST_SIZE * maxthreads + 1];
  if (max_value == 0) {
    return std::make_pair(inp_key_buf, inp_value_buf);
  }
  // If negative values are present, we want to perform all passes
  // up to a sign bit
  int num_bits = sizeof(K) * 8;
  if (!maybe_with_neg_vals)
    // __builtin_clz is not portable
    num_bits -= count_leading_zeros(static_cast<typename std::make_unsigned<K>::type>(max_value));
  const unsigned int num_passes = (num_bits + 7) / 8;

#pragma omp parallel
  {
    K* input_keys = inp_key_buf;
    V* input_values = inp_value_buf;
    K* output_keys = tmp_key_buf;
    V* output_values = tmp_value_buf;

    for (unsigned int pass = 0; pass < num_passes; pass++) {
      radix_sort_kernel(
          input_keys,
          input_values,
          output_keys,
          output_values,
          elements_count,
          histogram,
          histogram_ps,
          pass,
          maybe_with_neg_vals && pass == num_passes - 1);

      std::swap(input_keys, output_keys);
      std::swap(input_values, output_values);
#pragma omp barrier
    }
  }
  return (
      num_passes % 2 == 0 ? std::make_pair(inp_key_buf, inp_value_buf)
                          : std::make_pair(tmp_key_buf, tmp_value_buf));
}

} // namespace fbgemm
