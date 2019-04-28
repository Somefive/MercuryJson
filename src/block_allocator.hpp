#ifndef MERCURYJSON_BLOCK_ALLOCATOR_HPP
#define MERCURYJSON_BLOCK_ALLOCATOR_HPP

#include "flags.h"
#include "utils.h"


namespace MercuryJson {
    template <typename default_class>
    class BlockAllocator {
    private:
#if USE_BLOCK_ALLOCATOR
        char *mem, *ptr;
        std::vector<char *> all_memory;
        size_t block_size, allocated;
        static constexpr size_t alignment = sizeof(default_class);
#endif

#if USE_BLOCK_ALLOCATOR
        inline void check_alloc(size_t size) {
            if (allocated + size > block_size) {
                all_memory.push_back(mem);
                allocated = 0;
                ptr = mem = reinterpret_cast<char *>(aligned_malloc(block_size, alignment));
            }
        }
#endif

    public:
        explicit BlockAllocator(size_t block_size) {
#if USE_BLOCK_ALLOCATOR
            block_size = round_up(block_size, alignment);
            ptr = mem = reinterpret_cast<char *>(aligned_malloc(block_size, alignment));
            allocated = 0;
            this->block_size = block_size;
#endif
        }

        ~BlockAllocator() {
#if USE_BLOCK_ALLOCATOR
            // if (all_memory.size() > 0)
            //     printf("%lu blocks allocated\n", all_memory.size() + 1);
            aligned_free(mem);
            for (void *p : all_memory)
                aligned_free(p);
#endif
        }

        template <typename T = default_class>
        T *allocate(size_t size, size_t ensure_extra = 0) {
#if USE_BLOCK_ALLOCATOR
            size_t alloc_size = round_up(size * sizeof(T), alignment);
            check_alloc(alloc_size + ensure_extra);
            T *ret = new(ptr) T[size];
            ptr += alloc_size;
            allocated += alloc_size;
#else
            T *ret = new T[size + ensure_extra];
#endif
            return ret;
        }

        template <typename T = default_class, typename ...Args>
        T *construct(Args ...args) {
#if USE_BLOCK_ALLOCATOR
            check_alloc(sizeof(T));
            static constexpr size_t size = round_up(sizeof(T), alignment);
            T *ret = new(ptr) T(std::forward<Args>(args)...);
            ptr += size;
            allocated += size;
#else
            T *ret = new T(std::forward<Args>(args)...);
#endif
            return ret;
        }
    };
}

#endif //MERCURYJSON_BLOCK_ALLOCATOR_HPP
