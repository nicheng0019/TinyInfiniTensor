#include "operators/concat.h"
#include "utils/operator_utils.h"
#include "core/graph.h"
namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);

    // if (output == nullptr) {
    //     auto outputShape = inferShape(inputs);
    //     IT_ASSERT(outputShape.has_value());
    //     output = graph->addTensor(outputShape.value()[0], inputs[0]->getDType());
    //     outputs[0] = output;
    // }

    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // 计算在指定维度上所有输入张量的尺寸总和
    for (size_t i = 1; i < inputs.size(); ++i) {
        dims[dim] += inputs[i]->getDims()[dim];
    }
    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
