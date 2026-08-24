#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../nn/rope.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(RMSNormRoPE, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, float, infinicore::nn::RoPE::Algo);

// Internal: fused per-head RMSNorm + RoPE (full rotary), in-place on x
// x: [num_tokens, num_heads, head_dim]
void rms_norm_rope_(Tensor x,
                    const Tensor &weight,
                    const Tensor &pos_ids,
                    const Tensor &sin_table,
                    const Tensor &cos_table,
                    float epsilon,
                    infinicore::nn::RoPE::Algo algo);

} // namespace infinicore::op
