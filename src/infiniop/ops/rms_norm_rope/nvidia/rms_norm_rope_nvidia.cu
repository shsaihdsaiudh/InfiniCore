#include "../../../devices/nvidia/nvidia_common.cuh"
#include "rms_norm_rope_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../cuda/kernel.cuh"

template <unsigned int BLOCK_SIZE, bool IsGPTJ, typename Tdata, typename Tweight, typename Tangle, typename Tindex>
INFINIOP_CUDA_KERNEL rmsNormRopeKernel(
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
    rmsNormRopeBlock<BLOCK_SIZE, IsGPTJ>(x, stride_x_token, stride_x_head, w, pos_ids, pos_stride, sin_table, cos_table, table_dim, epsilon);
}

namespace op::rms_norm_rope::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc,
    float epsilon,
    infiniopRoPEAlgo_t algo) {
    auto result = RMSNormRoPEInfo::create(x_desc, w_desc, pos_desc, sin_desc, cos_desc, epsilon, algo);
    CHECK_RESULT(result);
    auto info = result.take();

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// launch kernel with different data types
// sin/cos tables always have the same dtype as x (checked in RMSNormRoPEInfo),
// so they are interpreted as Tdata, same as the rope kernel launch
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    uint32_t num_tokens, size_t nhead, size_t table_dim,
    void *x, infiniDtype_t atype, ptrdiff_t stride_x_token, ptrdiff_t stride_x_head,
    const void *w, infiniDtype_t wtype,
    const void *pos_ids, infiniDtype_t pos_type, ptrdiff_t pos_stride,
    const void *sin_table, const void *cos_table,
    float epsilon, infiniopRoPEAlgo_t algo,
    cudaStream_t cuda_stream) {

    dim3 grid_dim(num_tokens, uint32_t(nhead));

#define LAUNCH_KERNEL(Tdata, Tweight, Tindex, IsGPTJ)                                       \
    rmsNormRopeKernel<BLOCK_SIZE, IsGPTJ, Tdata, Tweight, Tdata, Tindex>                    \
        <<<grid_dim, BLOCK_SIZE, 0, cuda_stream>>>(                                         \
            reinterpret_cast<Tdata *>(x),                                                   \
            stride_x_token,                                                                 \
            stride_x_head,                                                                  \
            reinterpret_cast<const Tweight *>(w),                                           \
            reinterpret_cast<const Tindex *>(pos_ids),                                      \
            pos_stride,                                                                     \
            reinterpret_cast<const Tdata *>(sin_table),                                     \
            reinterpret_cast<const Tdata *>(cos_table),                                     \
            table_dim,                                                                      \
            epsilon)

#define DISPATCH_ALGO(Tdata, Tweight, Tindex)                \
    if (algo == INFINIOP_ROPE_ALGO_GPT_J) {                  \
        LAUNCH_KERNEL(Tdata, Tweight, Tindex, true);         \
    } else {                                                 \
        LAUNCH_KERNEL(Tdata, Tweight, Tindex, false);        \
    }                                                        \
    return INFINI_STATUS_SUCCESS

#define DISPATCH_POS(Tdata, Tweight)                         \
    if (pos_type == INFINI_DTYPE_I32) {                      \
        DISPATCH_ALGO(Tdata, Tweight, int32_t);              \
    } else if (pos_type == INFINI_DTYPE_I64) {               \
        DISPATCH_ALGO(Tdata, Tweight, int64_t);              \
    } else {                                                 \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;               \
    }

    if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F16) {
        DISPATCH_POS(half, half);
    } else if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_BF16) {
        DISPATCH_POS(half, __nv_bfloat16);
    } else if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F32) {
        DISPATCH_POS(half, float);
    } else if (atype == INFINI_DTYPE_BF16 && wtype == INFINI_DTYPE_BF16) {
        DISPATCH_POS(__nv_bfloat16, __nv_bfloat16);
    } else if (atype == INFINI_DTYPE_BF16 && wtype == INFINI_DTYPE_F16) {
        DISPATCH_POS(__nv_bfloat16, half);
    } else if (atype == INFINI_DTYPE_BF16 && wtype == INFINI_DTYPE_F32) {
        DISPATCH_POS(__nv_bfloat16, float);
    } else if (atype == INFINI_DTYPE_F32 && wtype == INFINI_DTYPE_F32) {
        DISPATCH_POS(float, float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef DISPATCH_POS
#undef DISPATCH_ALGO
#undef LAUNCH_KERNEL

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *x, const void *w,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // launch kernel with block size matching the number of rotation pairs
    if (_info.table_dim <= 64) {
        CHECK_STATUS(launchKernel<64>(uint32_t(_info.num_tokens), _info.num_heads, _info.table_dim, x, _info.atype, _info.x_stride_token, _info.x_stride_head, w, _info.wtype, pos_ids, _info.pos_type, _info.pos_stride, sin_table, cos_table, _info.epsilon, _info.algo, cuda_stream));
    } else if (_info.table_dim <= 128) {
        CHECK_STATUS(launchKernel<128>(uint32_t(_info.num_tokens), _info.num_heads, _info.table_dim, x, _info.atype, _info.x_stride_token, _info.x_stride_head, w, _info.wtype, pos_ids, _info.pos_type, _info.pos_stride, sin_table, cos_table, _info.epsilon, _info.algo, cuda_stream));
    } else {
        CHECK_STATUS(launchKernel<256>(uint32_t(_info.num_tokens), _info.num_heads, _info.table_dim, x, _info.atype, _info.x_stride_token, _info.x_stride_head, w, _info.wtype, pos_ids, _info.pos_type, _info.pos_stride, sin_table, cos_table, _info.epsilon, _info.algo, cuda_stream));
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::rms_norm_rope::nvidia
