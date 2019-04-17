#include <stdexcept>

#include <cstdio>
#include <cstdlib>

#include "utils.h"

char *read_file(const char *filename, size_t *size) {
    std::FILE *pfile = std::fopen(filename, "rb");
    if (pfile == nullptr) throw std::runtime_error("file open error");
    std::fseek(pfile, 0, SEEK_END);
    size_t _size = std::ftell(pfile);
    char *buffer = (char *)aligned_malloc(ALIGNMENT_SIZE, _size);
    if (buffer == nullptr) {
        std::fclose(pfile);
        throw std::runtime_error("allocate memory failed");
    }
    std::rewind(pfile);
    if (std::fread(buffer, 1, _size, pfile) != _size) {
        free(pfile);
        std::fclose(pfile);
        throw std::runtime_error("read data failed");
    }
    std::fclose(pfile);
    *size = _size;
    return buffer;
}
