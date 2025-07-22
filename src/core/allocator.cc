#include "core/allocator.h"
#include <utility>
#include <map>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size) {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        if (!freeBlocks.empty()) {
            auto lastBlock = freeBlocks.rbegin(); // Get the last block
            if (lastBlock->first + lastBlock->second == peak) {
                // This is an end block, check if we can use/extend it
                if (lastBlock->second >= size) {
                    // Use the end block
                    size_t addr = lastBlock->first;
                    size_t blockSize = lastBlock->second;
                    freeBlocks.erase(std::prev(freeBlocks.end()));
                    
                    if (blockSize > size) {
                        freeBlocks[addr + size] = blockSize - size;
                    }
                    
                    used += size;
                    return addr;
                } else {
                    // Extend the end block
                    size_t addr = lastBlock->first;
                    size_t additionalSize = size - lastBlock->second;
                    freeBlocks.erase(std::prev(freeBlocks.end()));
                    peak += additionalSize;
                    used += size;
                    return addr;
                }
            }
        }
        
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            if (it->second >= size) {
                size_t addr = it->first;
                size_t blockSize = it->second;
                
                // Remove the free block
                freeBlocks.erase(it);
                
                // If block is larger than needed, split it
                if (blockSize > size) {
                    freeBlocks[addr + size] = blockSize - size;
                }
                
                used += size;
                return addr;
            }
        }

        size_t addr = peak;
        peak += size;
        used += size;
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        used -= size;
        
        // Insert the freed block
        freeBlocks[addr] = size;
        
        // Merge with adjacent blocks
        auto it = freeBlocks.find(addr);
        
        // Merge with next block
        auto next = std::next(it);
        if (next != freeBlocks.end() && addr + size == next->first) {
            size += next->second;
            freeBlocks.erase(next);
            it->second = size;
        }
        
        // Merge with previous block
        if (it != freeBlocks.begin()) {
            auto prev = std::prev(it);
            if (prev->first + prev->second == addr) {
                prev->second += size;
                freeBlocks.erase(it);
            }
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
