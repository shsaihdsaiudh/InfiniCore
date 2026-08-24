#ifndef __INFINIOP_RMS_NORM_ROPE_API_H__
#define __INFINIOP_RMS_NORM_ROPE_API_H__

#include "../operator_descriptor.h"
#include "rope.h"

typedef struct InfiniopDescriptor *infiniopRMSNormRoPEDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateRMSNormRoPEDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t pos_ids_desc,
    infiniopTensorDescriptor_t sin_table_desc,
    infiniopTensorDescriptor_t cos_table_desc,
    float epsilon,
    infiniopRoPEAlgo_t algo);

__INFINI_C __export infiniStatus_t infiniopGetRMSNormRoPEWorkspaceSize(infiniopRMSNormRoPEDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopRMSNormRoPE(
    infiniopRMSNormRoPEDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *x,
    void const *weight,
    void const *pos_ids,
    void const *sin_table,
    void const *cos_table,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyRMSNormRoPEDescriptor(infiniopRMSNormRoPEDescriptor_t desc);

#endif
