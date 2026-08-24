#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/rms_norm_rope.h"

#ifdef ENABLE_CPU_API
#include "cpu/rms_norm_rope_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/rms_norm_rope_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateRMSNormRoPEDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t pos_ids_desc,
    infiniopTensorDescriptor_t sin_table_desc,
    infiniopTensorDescriptor_t cos_table_desc,
    float epsilon,
    infiniopRoPEAlgo_t algo) {

#define CREATE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                         \
        return op::rms_norm_rope::NAMESPACE::Descriptor::create(                       \
            handle,                                                                    \
            reinterpret_cast<op::rms_norm_rope::NAMESPACE::Descriptor **>(desc_ptr),   \
            x_desc,                                                                    \
            weight_desc,                                                               \
            pos_ids_desc,                                                              \
            sin_table_desc,                                                            \
            cos_table_desc,                                                            \
            epsilon,                                                                   \
            algo)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetRMSNormRoPEWorkspaceSize(infiniopRMSNormRoPEDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                           \
    case CASE:                                                                                         \
        *size = reinterpret_cast<op::rms_norm_rope::NAMESPACE::Descriptor *>(desc)->workspaceSize();   \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__INFINI_C infiniStatus_t infiniopRMSNormRoPE(
    infiniopRMSNormRoPEDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *x,
    void const *weight,
    void const *pos_ids,
    void const *sin_table,
    void const *cos_table,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                             \
    case CASE:                                                                                 \
        return reinterpret_cast<const op::rms_norm_rope::NAMESPACE::Descriptor *>(desc)        \
            ->calculate(workspace, workspace_size, x, weight, pos_ids, sin_table, cos_table, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyRMSNormRoPEDescriptor(infiniopRMSNormRoPEDescriptor_t desc) {
    if (desc == nullptr) {
        return INFINI_STATUS_SUCCESS;
    }

#define DESTROY(CASE, NAMESPACE)                                                     \
    case CASE:                                                                       \
        delete reinterpret_cast<op::rms_norm_rope::NAMESPACE::Descriptor *>(desc);   \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
