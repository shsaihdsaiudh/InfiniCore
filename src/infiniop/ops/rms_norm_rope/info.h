#ifndef __RMS_NORM_ROPE_INFO_H__
#define __RMS_NORM_ROPE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infiniop/ops/rope.h"

namespace op::rms_norm_rope {

class RMSNormRoPEInfo {
    RMSNormRoPEInfo() = default;

public:
    infiniDtype_t atype;
    infiniDtype_t wtype;
    infiniDtype_t pos_type;
    float epsilon;
    size_t num_tokens, num_heads, head_dim, table_len, table_dim;
    ptrdiff_t x_stride_token;
    ptrdiff_t x_stride_head;
    ptrdiff_t pos_stride;
    infiniopRoPEAlgo_t algo;

    static utils::Result<RMSNormRoPEInfo> create(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        infiniopTensorDescriptor_t pos_desc,
        infiniopTensorDescriptor_t sin_desc,
        infiniopTensorDescriptor_t cos_desc,
        float epsilon,
        infiniopRoPEAlgo_t algo) {

        CHECK_OR_RETURN(
            x_desc != nullptr && w_desc != nullptr && pos_desc != nullptr && sin_desc != nullptr && cos_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(algo < infiniopRoPEAlgo_t::INFINIOP_ROPE_ALGO_COUNT, INFINI_STATUS_BAD_PARAM);

        const infiniDtype_t atype = x_desc->dtype();
        const infiniDtype_t wtype = w_desc->dtype();
        const infiniDtype_t pos_type = pos_desc->dtype();

        if (atype == INFINI_DTYPE_F16 || atype == INFINI_DTYPE_BF16) {
            // For half-precision types (FP16/BF16), weights can be the same half-precision type or FP32
            if (wtype != atype && wtype != INFINI_DTYPE_F32 && wtype != INFINI_DTYPE_BF16 && wtype != INFINI_DTYPE_F16) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else if (atype == INFINI_DTYPE_F32) {
            // For FP32, activations and weights must be of the same type
            if (wtype != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        // Position IDs must be 32/64-bit integers
        CHECK_DTYPE(pos_type, INFINI_DTYPE_I32, INFINI_DTYPE_I64);
        // sin/cos tables must have the same dtype as x (same constraint as the rope op)
        CHECK_OR_RETURN(atype == sin_desc->dtype() && atype == cos_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);

        // x: [num_tokens, num_heads, head_dim], weight: [head_dim], pos_ids: [num_tokens]
        CHECK_OR_RETURN(x_desc->ndim() == 3 && w_desc->ndim() == 1 && pos_desc->ndim() == 1,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(sin_desc->ndim() == 2 && cos_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        const size_t num_tokens = x_desc->dim(0);
        const size_t num_heads = x_desc->dim(1);
        const size_t head_dim = x_desc->dim(2);

        CHECK_OR_RETURN(w_desc->dim(0) == head_dim && pos_desc->dim(0) == num_tokens,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        const auto table_len = sin_desc->dim(0);
        const auto table_dim = sin_desc->dim(1);
        CHECK_OR_RETURN(table_len == cos_desc->dim(0) && table_dim == cos_desc->dim(1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        // v1 only supports full rotary (rotary_dim == head_dim)
        CHECK_OR_RETURN(head_dim == table_dim * 2, INFINI_STATUS_BAD_TENSOR_SHAPE);

        // Last dimension of x must be contiguous, weight must be contiguous
        CHECK_OR_RETURN(x_desc->stride(2) == 1 && w_desc->stride(0) == 1,
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        // sin table and cos table must be totally contiguous
        CHECK_OR_RETURN(sin_desc->isContiguous() && cos_desc->isContiguous(),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<RMSNormRoPEInfo>(RMSNormRoPEInfo{
            atype,
            wtype,
            pos_type,
            epsilon,
            num_tokens,
            num_heads,
            head_dim,
            table_len,
            table_dim,
            x_desc->stride(0),
            x_desc->stride(1),
            pos_desc->stride(0),
            algo,
        });
    }
};

} // namespace op::rms_norm_rope

#endif // __RMS_NORM_ROPE_INFO_H__
