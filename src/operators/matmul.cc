#include "operators/matmul.h"
#include "utils/operator_utils.h"
namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        const auto B = inputs[1];
        auto shapeA = A->getDims();
        auto shapeB = B->getDims();
        
        IT_ASSERT(shapeA.size() >= 2 && shapeB.size() >= 2);
        
        // Get matrix dimensions (last 2 dimensions)
        int rankA = shapeA.size();
        int rankB = shapeB.size();
        
        // Apply transpose to get actual matrix dimensions
        int dimA_m = transA ? shapeA[rankA-1] : shapeA[rankA-2];
        int dimA_k = transA ? shapeA[rankA-2] : shapeA[rankA-1];
        int dimB_k = transB ? shapeB[rankB-1] : shapeB[rankB-2];
        int dimB_n = transB ? shapeB[rankB-2] : shapeB[rankB-1];
        
        // Check matrix multiplication compatibility
        IT_ASSERT(dimA_k == dimB_k);
        
        // Handle batch dimensions (all dimensions except last 2)
        Shape batchA(shapeA.begin(), shapeA.end() - 2);
        Shape batchB(shapeB.begin(), shapeB.end() - 2);
        Shape batchResult = infer_broadcast(batchA, batchB);
        
        // Construct output shape: batch dimensions + [m, n]
        Shape result = batchResult;
        result.push_back(dimA_m);
        result.push_back(dimB_n);
        
        return {{result}};
    }

} // namespace infini