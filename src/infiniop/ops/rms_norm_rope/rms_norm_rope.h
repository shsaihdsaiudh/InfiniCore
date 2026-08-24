#ifndef RMS_NORM_ROPE_H
#define RMS_NORM_ROPE_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::rms_norm_rope::NAMESPACE {                     \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        RMSNormRoPEInfo _info;                                   \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            RMSNormRoPEInfo info,                                \
            size_t workspace_size,                               \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t x_desc,                   \
            infiniopTensorDescriptor_t w_desc,                   \
            infiniopTensorDescriptor_t pos_desc,                 \
            infiniopTensorDescriptor_t sin_desc,                 \
            infiniopTensorDescriptor_t cos_desc,                 \
            float epsilon,                                       \
            infiniopRoPEAlgo_t algo);                            \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *x,                                             \
            const void *w,                                       \
            const void *pos_ids,                                 \
            const void *sin_table,                               \
            const void *cos_table,                               \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // RMS_NORM_ROPE_H
