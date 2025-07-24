#pragma once
#include "core/allocator.h"
#include "core/operator.h"
#include "core/tensor.h"
#include <algorithm>
#include <cstdint>

namespace infini
{

    class GraphObj : public Object
    {
    protected:
        Runtime runtime;
        TensorVec tensors;
        OpVec ops;
        Allocator allocator;

    public:
        explicit GraphObj(Runtime runtime)
            : runtime(runtime), allocator(runtime), sorted(false){};
        string toString() const override;
        Runtime getRuntime() const { return runtime; }

        Tensor addTensor(Shape dim, DataType dtype = DataType::Float32);
        Tensor addTensor(const Tensor &tensor);
        TensorVec addTensor(const TensorVec &tensors);
        void removeOperator(Operator op)
        {
            auto it = std::find(ops.begin(), ops.end(), op);
            if (it != ops.end())
                ops.erase(it);
        }

        void removeTensor(Tensor tensor)
        {
            auto it = std::find(tensors.begin(), tensors.end(), tensor);
            if (it != tensors.end())
                tensors.erase(it);
        }

        const TensorVec &getTensors() const { return tensors; }
        const OpVec &getOperators() const { return ops; }
        Tensor getTensor(int) const;

        /**
         * @brief Sort the nodes in topological order.
         * It returns true if the sorting is successful.
         * Otherwise false is returned, means that there are rings in the graph,
         * so the topological sorting fails.
         */
        bool topo_sort();

        void optimize();

        void shape_infer();

        void dataMalloc();

        /**
         * @brief Add an operator and create its outputs. Output tensor arguments
         * should be empty Refs (e.g., nullptr).
         */
        template <typename T, typename... Args>
        Ref<T> addOp(Args &&...args)
        {
            Ref<T> op = infini::make_ref<T>(this, std::forward<Args>(args)...);
            addOperatorAndConnect(op);
            return op;
        }

        /**
         * @brief Add an operator with its outputs specified.
         */
        template <typename T, typename... Args>
        Ref<T> addOpWithOutputs(Args &&...args)
        {
            Ref<T> op = infini::make_ref<T>(nullptr, std::forward<Args>(args)...);
            addOperatorAndConnect(op);
            return op;
        }

        /**
         * @brief Gets input tensors of this graph.
         */
        inline TensorVec getInputs() const
        {
            TensorVec ret;
            for (const auto &t : tensors)
                if (!t->getSource())
                    ret.emplace_back(t);
            return ret;
        }

        /**
         * @brief Gets output tensors of this graph.
         */
        inline TensorVec getOutputs() const
        {
            TensorVec ret;
            for (const auto &t : tensors)
                if (t->getTargets().empty())
                    ret.emplace_back(t);
            return ret;
        }

        bool checkValid() const;

    private:
        /**
         * @brief Add reverse connections and Op relationship in ctor.
         */
        void addOperatorAndConnect(const Operator &op);

        /**
         * @brief Remove redundant transpose operators that are inverse of each other.
         * @return true if any optimization was performed, false otherwise.
         */
         bool removeRedundantTranspose();

         /**
          * @brief Fuse transpose operations into matmul operators by setting transA/transB flags.
          * @return true if any optimization was performed, false otherwise.
          */
         bool fuseTransposeIntoMatmul();
 
         /**
          * @brief Check if two permutation vectors are inverse of each other.
          * @param perm1 First permutation vector
          * @param perm2 Second permutation vector
          * @return true if perm2 is the inverse of perm1
          */
         bool isInversePermutation(const std::vector<int>& perm1, const std::vector<int>& perm2);
 
         /**
          * @brief Check if a permutation represents a transpose of the last two dimensions.
          * @param permute The permutation vector
          * @param rank The rank of the tensor
          * @return true if it's a last-two-dimension transpose
          */
         bool isLastTwoDimTranspose(const std::vector<int>& permute, int rank);
 
         /**
          * @brief Reconnect tensors by redirecting all targets from one tensor to another.
          * @param from Source tensor
          * @param to Target tensor
          */
         void reconnectTensors(Tensor from, Tensor to);
 
         /**
          * @brief Replace an operator in the graph with a new operator.
          * @param oldOp The operator to be replaced
          * @param newOp The new operator to replace with
          */
         void replaceOperator(Operator oldOp, Operator newOp);

         bool removeOperatorfromGraph(Operator op);
        /**
         * @brief If the nodes is sorted in topological order.
         */
        bool sorted;
    };

} // namespace infini
