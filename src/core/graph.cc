#include "core/graph.h"
#include "operators/transpose.h"
#include "operators/matmul.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";

        for (const auto &tensor : tensors)
        {
            oss << tensor << "\n";
        }
            
        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
            {
                preds.emplace_back(o->getGuid());
            }
                
            for (auto &o : op->getSuccessors())
            {
                succs.emplace_back(o->getGuid());
            }
                
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

   
    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
    
        bool optimized = true;
        while (optimized) {
            optimized = false;
            
            // 规则1: 去除冗余的transpose算子
            optimized |= removeRedundantTranspose();
            // 规则2: 将transpose融入matmul算子
            optimized |= fuseTransposeIntoMatmul();
            
        }
        
        // 重新标记需要拓扑排序
        sorted = false;
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        std::vector<size_t> offsets;
        for (auto &tensor : tensors) {
            // 获取tensor需要的内存大小
            size_t tensorSize = tensor->getBytes();
            
            // 使用allocator分配内存，获取偏移地址
            size_t offset = allocator.alloc(tensorSize);
            offsets.push_back(offset);
        }

        // 为每个tensor分配内存
        for (size_t i = 0; i < tensors.size(); i++) {
            auto &tensor = tensors[i];
            size_t offset = offsets[i];
            
            // 获取实际的内存指针
            void *basePtr = allocator.getPtr();
            
            // 计算tensor的实际内存地址
            void *tensorPtr = static_cast<char*>(basePtr) + offset;
            
            // 创建Blob并绑定到tensor
            auto blob = make_ref<BlobObj>(runtime, tensorPtr);
            tensor->setDataBlob(blob);
        }
        
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

    bool GraphObj::removeOperatorfromGraph(Operator op)
    {
        for (auto &o : op->getPredecessors())
        {
            o->removeSuccessors(op);
        } 

        for (auto &o : op->getSuccessors())
        {
            o->removePredecessors(op);
        }
        removeOperator(op);
        return true;
    }

    bool GraphObj::removeRedundantTranspose()
    {
        bool changed = false;

        // 使用索引遍历避免迭代器失效
        for (size_t i = 0; i < ops.size();) {
            auto op = ops[i];

            // 检查是否为transpose算子
            if (op->getOpType() != OpType::Transpose) {
                ++i;
                continue;
            }

            auto transposeOp = as<TransposeObj>(op);
            auto output = transposeOp->getOutput();

            // 检查是否只有一个后继且也是transpose
            if (output->getTargets().size() == 1) {
                auto nextOp = output->getTargets()[0];
                if (nextOp->getOpType() == OpType::Transpose) {
                    auto nextTranspose = as<TransposeObj>(nextOp);

                    // 检查两个transpose是否互为逆操作
                    auto perm1 = transposeOp->getPermute();
                    auto perm2 = nextTranspose->getPermute();

                    if (isInversePermutation(perm1, perm2)) {
                        // 移除这两个transpose算子
                        auto input = transposeOp->getInputs()[0];
                        auto finalOutput = nextTranspose->getOutput();

                        // 重新连接输入到最终输出
                        reconnectTensors(input, finalOutput);

                        // 从图中移除算子和中间tensor
                   
                        removeOperatorfromGraph(op);
                        removeOperatorfromGraph(nextOp);
                        removeTensor(output);
                        removeTensor(finalOutput);

                        changed = true;
                        i = 0; // 重新开始遍历
                        continue;
                    }
                }
            }
            ++i;
        }

        topo_sort();

        return changed;
    }

    bool GraphObj::fuseTransposeIntoMatmul()
    {
        bool changed = false;

        // 使用索引遍历避免迭代器失效
        for (size_t i = 0; i < ops.size(); ++i) {
            auto op = ops[i];
            if (op->getOpType() != OpType::MatMul) {
                continue;
            }

            auto matmulOp = as<MatmulObj>(op);
            auto inputA = matmulOp->getInputs()[0];
            auto inputB = matmulOp->getInputs()[1];

            bool fusedA = false, fusedB = false;

            // 检查输入A是否来自transpose
            if (auto sourceA = inputA->getSource()) {
                if (sourceA->getOpType() == OpType::Transpose) {
                    auto transposeA = as<TransposeObj>(sourceA);
                    if (isLastTwoDimTranspose(transposeA->getPermute(), inputA->getRank())) {
                        // 融入transA属性
                        auto newMatmul = make_ref<MatmulObj>(nullptr,
                            transposeA->getInputs()[0], inputB, matmulOp->getOutput(),
                            !matmulOp->getTransA(), matmulOp->getTransB());

                        replaceOperator(matmulOp, newMatmul);

                        // 只有当transpose没有其他用户时才移除
                        if (inputA->getTargets().size() == 1) {
                            removeOperatorfromGraph(sourceA);
                            removeTensor(inputA);
                        }

                        fusedA = true;
                        changed = true;
                    }
                }
            }

            auto sourceB = inputB->getSource();
            // 检查输入B是否来自transpose（如果A没有被融合）
            if (!fusedA &&  sourceB) {
                if (sourceB->getOpType() == OpType::Transpose) {
                    auto transposeB = as<TransposeObj>(sourceB);
                    if (isLastTwoDimTranspose(transposeB->getPermute(), inputB->getRank())) {
                        // 融入transB属性
                        
                        auto inputtransposeB = transposeB->getInputs()[0];
                        inputtransposeB->removeTarget(transposeB);
                        auto newMatmul = make_ref<MatmulObj>(nullptr,
                            inputA, inputtransposeB, matmulOp->getOutput(),
                            matmulOp->getTransA(), !matmulOp->getTransB());
                                              
                        inputtransposeB->addTarget(newMatmul);
                        replaceOperator(matmulOp, newMatmul);
                                              
                        // 只有当transpose没有其他用户时才移除
                        if (inputB->getTargets().size() == 1) {
                            
                            removeOperatorfromGraph(sourceB);
                            removeTensor(inputB);
                        }
                        
                        changed = true;
                    }
                }
            }
        }

        return changed;
    }

    bool GraphObj::isInversePermutation(const vector<int>& perm1, const vector<int>& perm2)
    {
        if (perm1.size() != perm2.size()) return false;
        
        vector<int> composed(perm1.size());
        for (size_t i = 0; i < perm1.size(); ++i) {
            composed[i] = perm2[perm1[i]];
        }
        
        // 检查是否为恒等置换
        for (size_t i = 0; i < composed.size(); ++i) {
            if (composed[i] != (int)i) return false;
        }
        return true;
    }

    bool GraphObj::isLastTwoDimTranspose(const vector<int>& perm, int rank)
    {
        if (rank < 2) return false;
        
        // 检查是否只交换最后两个维度
        for (int i = 0; i < rank - 2; ++i) {
            if (perm[i] != i) return false;
        }
        
        return perm[rank-2] == rank-1 && perm[rank-1] == rank-2;
    }

    void GraphObj::reconnectTensors(Tensor from, Tensor to)
    {
        // 复制目标列表避免在迭代时修改
        auto targets = to->getTargets();
        for (auto target : from->getTargets())
        {
            from->removeTarget(target);
        }
        // 将from的所有目标重定向到to
        for (auto target : targets) {
            target->replaceInput(to, from);
            from->addTarget(target);
            to->removeTarget(target);
        }

    }

    void GraphObj::replaceOperator(Operator oldOp, Operator newOp)
    {
        // 替换ops列表中的算子
        auto it = std::find(ops.begin(), ops.end(), oldOp);
        if (it != ops.end()) {
            *it = newOp;
        }

        // 更新输入tensor的目标连接
        for (auto input : oldOp->getInputs()) {
            if (input) {
                input->removeTarget(oldOp);
                input->addTarget(newOp);
                if (auto source = input->getSource()) {
                    source->addSuccessors(newOp);
                    newOp->addPredecessors(source);
                }
            }
        }

        // 更新输出tensor的source
        for (auto output : oldOp->getOutputs()) {
            if (output) {
                output->setSource(newOp);
                for (auto target : output->getTargets()) {
                    target->addPredecessors(newOp);
                    newOp->addSuccessors(target);
                }
            }
        }
    }

} // namespace infini
