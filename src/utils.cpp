#include "utils.h"

#include <stdexcept>

#include <stdio.h>
#include <stdlib.h>


char *read_file(const char *filename, size_t *size) {
    std::FILE *pfile = std::fopen(filename, "rb");
    if (pfile == nullptr) throw std::runtime_error("file open error");
    std::fseek(pfile, 0, SEEK_END);
    size_t _size = std::ftell(pfile);
    char *buffer = aligned_malloc(_size + 2 * kAlignmentSize);
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
    buffer[_size] = '\0';
    *size = _size;
    return buffer;
}

void print_indent(int indent) {
    if (indent > 0) {
        std::cout << std::string(indent, ' ');
    }
}

double plain_convert(long long int value) {
    union { long long int integer; double decimal; };
    integer = value; return decimal;
}

long long int plain_convert(double value) {
    union { long long int integer; double decimal; };
    decimal = value; return integer;
}
