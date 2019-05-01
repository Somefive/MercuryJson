#ifndef MERCURYJSON_BLOCK_ALLOCATOR_HPP
#define MERCURYJSON_BLOCK_ALLOCATOR_HPP

#include <stdexcept>
#include <vector>

#include "flags.h"
#include "utils.h"


namespace MercuryJson {

    template <typename default_class>
    class BlockAllocator {
    private:
        static constexpr size_t kAlignment = sizeof(default_class);

        char *ptr;
        std::vector<char *> all_memory;
        size_t block_size, allocated;
        BlockAllocator *parent;

        inline void check_alloc(size_t size) {
            if (allocated + size > block_size) {
                allocated = 0;
                ptr = reinterpret_cast<char *>(aligned_malloc(block_size, kAlignment));
                if (parent != nullptr) parent->all_memory.push_back(ptr);  // TODO: Make this thread-safe.
                else all_memory.push_back(ptr);
            }
        }

        BlockAllocator(size_t block_size, BlockAllocator *parent) : parent(parent) {
            block_size = round_up(block_size, kAlignment);
            ptr = reinterpret_cast<char *>(aligned_malloc(block_size, kAlignment));
            parent->all_memory.push_back(ptr);
            allocated = 0;
            this->block_size = block_size;
        }

    public:
        explicit BlockAllocator(size_t block_size) : parent(nullptr) {
            block_size = round_up(block_size, kAlignment);
            ptr = reinterpret_cast<char *>(aligned_malloc(block_size, kAlignment));
            all_memory.push_back(ptr);
            allocated = 0;
            this->block_size = block_size;
        }

        BlockAllocator(BlockAllocator &&other) noexcept {
            ptr = other.ptr;
            all_memory = std::move(other.all_memory);
            block_size = other.block_size;
            allocated = other.allocated;
            parent = other.parent;
        }

        ~BlockAllocator() {
            for (char *p : all_memory)
                aligned_free(p);
        }

        BlockAllocator fork() {
            if (parent != nullptr) throw std::runtime_error("Cannot fork a forked allocator.");
            return {block_size, this};
        }

        BlockAllocator fork(size_t block_size) {
            if (parent != nullptr) throw std::runtime_error("Cannot fork a forked allocator.");
            return {block_size, this};
        }

        inline size_t size() { return block_size; }

        template <typename T = default_class>
        T *allocate(size_t size, size_t ensure_extra = 0) {
            size_t alloc_size = round_up(size * sizeof(T), kAlignment);
            check_alloc(alloc_size + ensure_extra);
            T *ret = new(ptr) T[size];
            ptr += alloc_size;
            allocated += alloc_size;
            return ret;
        }

        template <typename T = default_class, typename ...Args>
        T *construct(Args ...args) {
            check_alloc(sizeof(T));
            static constexpr size_t size = round_up(sizeof(T), kAlignment);
            T *ret = new(ptr) T(std::forward<Args>(args)...);
            ptr += size;
            allocated += size;
            return ret;
        }
    };

}

#endif // MERCURYJSON_BLOCK_ALLOCATOR_HPP
