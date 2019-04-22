#ifndef MERCURYJSON_UTILS_H
#define MERCURYJSON_UTILS_H

#include <cstdio>
#include <cstdlib>

static const size_t ALIGNMENT_SIZE = 64;

char *read_file(const char *filename, size_t *size);

inline constexpr size_t round_up(size_t size, size_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

static inline void *aligned_malloc(size_t alignment, size_t size) {
    void *p;
    size = round_up(size, alignment);
    if (posix_memalign(&p, alignment, size) != 0) { return nullptr; }
    return p;
}

template <typename T>
static inline void aligned_free(T *memblock) {
    if (memblock == nullptr) { return; }
    free(reinterpret_cast<void *>(memblock));
}

template <typename T>
static inline void aligned_free(const T *memblock) {
    aligned_free<T>(const_cast<T *>(memblock));
}

#endif // MERCURYJSON_UTILS_H
