#ifndef MERCURY_PARSESTRING_H
#define MERCURY_PARSESTRING_H

#include <cstddef>

namespace MercuryJson {

    void parse_str_per_bit(const char *src, char *dest, size_t *len = nullptr, size_t offset = 0U);
    void parse_str_naive(const char *src, char *dest = nullptr, size_t *len = nullptr, size_t offset = 0U);
    void parse_str_avx(const char *src, char *dest = nullptr, size_t *len = nullptr, size_t offset = 0U);
    void parse_str_none(const char *src, char *dest = nullptr, size_t *len = nullptr, size_t offset = 0U);

    #if PARSE_STR_MODE == 0
        #define parse_str parse_str_naive
    #elif PARSE_STR_MODE == 1
        #define parse_str parse_str_avx
    #elif PARSE_STR_MODE == 2
        #define parse_str parse_str_per_bit
    #else
        #define parse_str parse_str_none
    #endif

}

#endif