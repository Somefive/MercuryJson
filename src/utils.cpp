#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "utils.h"

char * read_file(const char * filename, size_t & size) {
    std::FILE * pfile = std::fopen(filename, "rb");
    if (pfile == NULL) throw std::runtime_error("file open error");
    std::fseek(pfile, 0, SEEK_END);
    size = std::ftell(pfile);
    char * buffer = (char *) aligned_alloc(ALIGNMENT_SIZE, size);
    if (buffer == NULL) {
        std::fclose(pfile);
        throw std::runtime_error("allocate memory failed");
    }
    std::rewind(pfile);
    if (std::fread(buffer, 1, size, pfile) != size) {
        free(pfile);
        std::fclose(pfile);
        throw std::runtime_error("read data failed");
    }
    std::fclose(pfile);
    return buffer;
}