#ifndef MERCURYJSON_UTILS_H
#define MERCURYJSON_UTILS_H

#include <iostream>

#include <stdio.h>
#include <stdlib.h>


static const size_t kAlignmentSize = 64;

char *read_file(const char *filename, size_t *size);

inline constexpr size_t round_up(size_t size, size_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

template <typename T = char>
static inline T *aligned_malloc(size_t size, size_t alignment = kAlignmentSize) {
    void *p;
    size = round_up(size * sizeof(T), alignment);
    if (posix_memalign(&p, alignment, size) != 0) { return nullptr; }
    return static_cast<T *>(p);
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

void print_indent(int indent);

double plain_convert(long long int value);
long long int plain_convert(double value);

#endif // MERCURYJSON_UTILS_H
