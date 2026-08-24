#include "rms_norm_rope_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::rms_norm_rope::cpu {

Descriptor::~Descriptor() {}

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
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tweight, typename Tindex>
infiniStatus_t rms_norm_rope(const RMSNormRoPEInfo *info,
                             Tdata *x, const Tweight *w, const Tindex *pos_ids,
                             const Tdata *sin_table, const Tdata *cos_table) {
    const size_t num_tokens = info->num_tokens;
    const size_t nhead = info->num_heads;
    const size_t table_dim = info->table_dim;
    const size_t dim = 2 * table_dim;
    const ptrdiff_t total_blocks = static_cast<ptrdiff_t>(num_tokens * nhead);

#pragma omp parallel for
    for (ptrdiff_t block_idx = 0; block_idx < total_blocks; ++block_idx) {
        const size_t tok = block_idx / nhead; // token index
        const size_t h = block_idx % nhead;   // head index

        Tdata *x_ptr = x + tok * info->x_stride_token + h * info->x_stride_head;

        const size_t pos_id = size_t(pos_ids[tok * info->pos_stride]);
        const Tdata *sin_ptr = sin_table + pos_id * table_dim;
        const Tdata *cos_ptr = cos_table + pos_id * table_dim;

        // [Reduce] sum of x^2 on the row
        float ss = 0.f;
        for (size_t k = 0; k < dim; k++) {
            float v = utils::cast<float>(x_ptr[k]);
            ss += v * v;
        }

        // 1 / (sqrt(sum/dim + eps))
        float rms = 1.f / std::sqrt(ss / (float)(dim) + info->epsilon);

        for (size_t i = 0; i < table_dim; i++) {
            // Calculate positions based on algorithm
            size_t pos0, pos1;
            if (info->algo == infiniopRoPEAlgo_t::INFINIOP_ROPE_ALGO_GPT_J) {
                // GPT-J style: interleaved pairs
                pos0 = 2 * i;
                pos1 = 2 * i + 1;
            } else {
                // GPT-NeoX style: first half and second half
                pos0 = i;
                pos1 = i + table_dim;
            }

            // Round normalized values to the storage dtype first
            // (mimicking the two-kernel pipeline), then rotate in fp32
            // with the same conversion and rounding order as the rope cpu kernel
            float x0 = utils::cast<float>(utils::cast<Tdata>(utils::cast<float>(x_ptr[pos0]) * utils::cast<float>(w[pos0]) * rms));
            float x1 = utils::cast<float>(utils::cast<Tdata>(utils::cast<float>(x_ptr[pos1]) * utils::cast<float>(w[pos1]) * rms));
            float sin__ = utils::cast<float>(sin_ptr[i]);
            float cos__ = utils::cast<float>(cos_ptr[i]);

            x_ptr[pos0] = utils::cast<Tdata>(x0 * cos__ - x1 * sin__);
            x_ptr[pos1] = utils::cast<Tdata>(x0 * sin__ + x1 * cos__);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE(Tdata, Tweight, Tindex) \
    rms_norm_rope(&_info, (Tdata *)x, (const Tweight *)w, (const Tindex *)pos_ids, (const Tdata *)sin_table, (const Tdata *)cos_table)

#define DISPATCH_POS(Tdata, Tweight)                                   \
    if (_info.pos_type == INFINI_DTYPE_I32) {                          \
        return CALCULATE(Tdata, Tweight, int32_t);                     \
    } else if (_info.pos_type == INFINI_DTYPE_I64) {                   \
        return CALCULATE(Tdata, Tweight, int64_t);                     \
    } else {                                                           \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;                         \
    }

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *x, const void *w,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    void *stream) const {
    if (_info.atype == INFINI_DTYPE_F16) {
        if (_info.wtype == INFINI_DTYPE_F16) {
            DISPATCH_POS(fp16_t, fp16_t);
        } else if (_info.wtype == INFINI_DTYPE_BF16) {
            DISPATCH_POS(fp16_t, bf16_t);
        } else if (_info.wtype == INFINI_DTYPE_F32) {
            DISPATCH_POS(fp16_t, float);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_info.atype == INFINI_DTYPE_BF16) {
        if (_info.wtype == INFINI_DTYPE_BF16) {
            DISPATCH_POS(bf16_t, bf16_t);
        } else if (_info.wtype == INFINI_DTYPE_F16) {
            DISPATCH_POS(bf16_t, fp16_t);
        } else if (_info.wtype == INFINI_DTYPE_F32) {
            DISPATCH_POS(bf16_t, float);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_info.atype == INFINI_DTYPE_F32) {
        DISPATCH_POS(float, float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

#undef DISPATCH_POS
#undef CALCULATE

} // namespace op::rms_norm_rope::cpu
