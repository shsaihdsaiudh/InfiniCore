#ifndef __RMS_NORM_ROPE_CUDA_KERNEL_CUH__
#define __RMS_NORM_ROPE_CUDA_KERNEL_CUH__

#include <cub/block/block_reduce.cuh>

// Fused per-head RMSNorm + RoPE, in-place on x.
// Each block takes care of one head in one token.
// Each thread deals with every BLOCK_SIZE-th rotation pair in the row;
// pair i covers elements (2i, 2i+1) for GPT-J and (i, i + table_dim) for GPT-NeoX,
// so the pairs partition the whole row for both algorithms.
template <unsigned int BLOCK_SIZE, bool IsGPTJ, typename Tdata, typename Tweight, typename Tangle, typename Tindex>
__device__ void rmsNormRopeBlock(
    Tdata *__restrict__ x,
    ptrdiff_t stride_x_token,
    ptrdiff_t stride_x_head,
    const Tweight *__restrict__ w,
    const Tindex *__restrict__ pos_ids,
    ptrdiff_t pos_stride,
    const Tangle *__restrict__ sin_table,
    const Tangle *__restrict__ cos_table,
    size_t table_dim,
    float epsilon) {

    size_t token_idx = blockIdx.x;
    size_t head_idx = blockIdx.y;

    auto x_ptr = x + token_idx * stride_x_token + head_idx * stride_x_head;

    size_t pos_id = size_t(pos_ids[token_idx * pos_stride]);
    auto sin_ptr = sin_table + pos_id * table_dim;
    auto cos_ptr = cos_table + pos_id * table_dim;

    // [Reduce] sum of x^2 on the row (fp32 accumulate, same as the rms_norm kernel)
    float sum_squared = 0;
    for (size_t i = threadIdx.x; i < table_dim; i += BLOCK_SIZE) {
        size_t pos0 = IsGPTJ ? 2 * i : i;
        size_t pos1 = IsGPTJ ? 2 * i + 1 : i + table_dim;
        float x0 = float(x_ptr[pos0]);
        float x1 = float(x_ptr[pos1]);
        sum_squared += x0 * x0 + x1 * x1;
    }

    // Block-reduce sum of squares
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum_squared = BlockReduce(temp_storage).Sum(sum_squared);

    // Thread_0 computes RMS=1/sqrt(ss/dim+epsilon) and stores in shared memory
    __shared__ float rms;
    if (threadIdx.x == 0) {
        rms = rsqrtf(sum_squared / float(2 * table_dim) + epsilon);
    }
    __syncthreads();

    // Normalize and round to the storage dtype first: n is bit-identical to the
    // rms_norm output, i.e. exactly what the rope kernel would read in the
    // two-kernel pipeline. Then rotate with the same per-branch arithmetic and
    // rounding points as the rope kernel (Tangle == Tdata, so e.g. the bf16
    // operators round back to bf16 after every operation).
    // Each pair is owned by a single thread, so reloading x here still reads
    // the original values.
    for (size_t i = threadIdx.x; i < table_dim; i += BLOCK_SIZE) {
        size_t pos0 = IsGPTJ ? 2 * i : i;
        size_t pos1 = IsGPTJ ? 2 * i + 1 : i + table_dim;

        Tdata n0 = Tdata(float(x_ptr[pos0]) * float(w[pos0]) * rms);
        Tdata n1 = Tdata(float(x_ptr[pos1]) * float(w[pos1]) * rms);
        Tangle sin__ = sin_ptr[i],
               cos__ = cos_ptr[i];

        if constexpr (IsGPTJ) {
            if constexpr (std::is_same<Tdata, half>::value) {
                // Same as the half2 path of the rope kernel: packed pair,
                // per-op rounding in half arithmetic
                half2 x_pair = half2(n0, n1);
                Tangle y0 = x_pair.x * cos__ - x_pair.y * sin__,
                       y1 = x_pair.x * sin__ + x_pair.y * cos__;
                half2 y_pair = half2(y0, y1);
                x_ptr[pos0] = y_pair.x;
                x_ptr[pos1] = y_pair.y;
            } else if constexpr (std::is_same<Tdata, cuda_bfloat16>::value) {
                // Same as the bfloat162 path of the rope kernel
                cuda_bfloat162 x_pair = cuda_bfloat162(n0, n1);
                Tangle x0 = __low2bfloat16(x_pair);
                Tangle x1 = __high2bfloat16(x_pair);
                Tangle y0 = x0 * cos__ - x1 * sin__;
                Tangle y1 = x0 * sin__ + x1 * cos__;
                cuda_bfloat162 y_pair = __floats2bfloat162_rn(y0, y1);
                x_ptr[pos0] = __low2bfloat16(y_pair);
                x_ptr[pos1] = __high2bfloat16(y_pair);
            } else {
                Tangle x0 = Tangle(n0),
                       x1 = Tangle(n1);
                x_ptr[pos0] = Tdata(x0 * cos__ - x1 * sin__);
                x_ptr[pos1] = Tdata(x0 * sin__ + x1 * cos__);
            }
        } else {
            if constexpr (std::is_same<Tdata, half>::value) {
                Tangle x0 = __half2float(n0);
                Tangle x1 = __half2float(n1);
                Tangle y0 = x0 * cos__ - x1 * sin__;
                Tangle y1 = x0 * sin__ + x1 * cos__;
                x_ptr[pos0] = __float2half(y0);
                x_ptr[pos1] = __float2half(y1);
            } else if constexpr (std::is_same<Tdata, cuda_bfloat16>::value) {
                Tangle x0 = __bfloat162float(n0);
                Tangle x1 = __bfloat162float(n1);
                Tangle y0 = x0 * cos__ - x1 * sin__;
                Tangle y1 = x0 * sin__ + x1 * cos__;
                x_ptr[pos0] = __float2bfloat16(y0);
                x_ptr[pos1] = __float2bfloat16(y1);
            } else {
                Tangle x0 = Tangle(n0),
                       x1 = Tangle(n1);
                x_ptr[pos0] = x0 * cos__ - x1 * sin__;
                x_ptr[pos1] = x0 * sin__ + x1 * cos__;
            }
        }
    }
}

#endif
