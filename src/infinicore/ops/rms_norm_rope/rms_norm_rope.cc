#include "infinicore/ops/rms_norm_rope.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(RMSNormRoPE);

RMSNormRoPE::RMSNormRoPE(Tensor x,
                         const Tensor &weight,
                         const Tensor &pos_ids,
                         const Tensor &sin_table,
                         const Tensor &cos_table,
                         float epsilon,
                         infinicore::nn::RoPE::Algo algo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, weight, pos_ids, sin_table, cos_table);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), x, weight, pos_ids, sin_table, cos_table, epsilon, algo);
}

void RMSNormRoPE::execute(Tensor x,
                          const Tensor &weight,
                          const Tensor &pos_ids,
                          const Tensor &sin_table,
                          const Tensor &cos_table,
                          float epsilon,
                          infinicore::nn::RoPE::Algo algo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(RMSNormRoPE, x, weight, pos_ids, sin_table, cos_table, epsilon, algo);
}

void rms_norm_rope_(Tensor x,
                    const Tensor &weight,
                    const Tensor &pos_ids,
                    const Tensor &sin_table,
                    const Tensor &cos_table,
                    float epsilon,
                    infinicore::nn::RoPE::Algo algo) {
    RMSNormRoPE::execute(x, weight, pos_ids, sin_table, cos_table, epsilon, algo);
}

} // namespace infinicore::op
