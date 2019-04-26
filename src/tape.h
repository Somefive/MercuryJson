#ifndef MERCURYJSON_H
#define MERCURYJSON_H

#include <cstdio>
#include <immintrin.h>
#include "utils.h"
#include "mercuryparser.h"

namespace MercuryJson {

    #define TYPE_MASK   0xf
    #define TYPE_NULL   0xf
    #define TYPE_FALSE  0x0
    #define TYPE_TRUE   0x1
    #define TYPE_STR    0x2
    #define TYPE_INT    0x3
    #define TYPE_DEC    0x4
    #define TYPE_OBJ    0x5
    #define TYPE_ARR    0x6

    struct Tape
    {
        u_int64_t *tape;
        char *literals;
        size_t tape_size, literals_size;
        Tape(size_t string_size, size_t structual_size) {
            tape = static_cast<u_int64_t *>(aligned_malloc(ALIGNMENT_SIZE, 2 * sizeof(u_int64_t) * (structual_size + ALIGNMENT_SIZE)));
            literals = static_cast<char *>(aligned_malloc(ALIGNMENT_SIZE, string_size + ALIGNMENT_SIZE));
            tape_size = 0;
            literals_size = 0;
        }
        ~Tape() {
            delete [] tape;
            delete [] literals;
        }
        void write_null() { tape[tape_size++] = TYPE_NULL; }
        void write_true() { tape[tape_size++] = TYPE_TRUE; }
        void write_false() { tape[tape_size++] = TYPE_FALSE; }
        void write_integer(long long int value) { tape[tape_size++] = TYPE_INT; tape[tape_size++] = value; }
        void write_decimal(double value) { tape[tape_size++] = TYPE_DEC; tape[tape_size++] = plain_convert(value); }
        size_t write_str(u_int64_t literal_idx) { tape[tape_size] = TYPE_STR | (literal_idx << 4); return tape_size++; }
        size_t write_array() { tape[tape_size] = TYPE_ARR; return tape_size++; }
        size_t write_object() { tape[tape_size] = TYPE_OBJ; return tape_size++; }
        void write_content(u_int64_t content, size_t idx) { tape[idx] = (tape[idx] & TYPE_MASK) | (content << 4); }
        size_t print_json(size_t tape_idx=0, size_t indent=0);
    };

    struct TapeWriter {
        Tape *tape;
        char *input;
        size_t *idxptr;
        TapeWriter(Tape *_tape, char *_input, size_t *_idxptr): tape(_tape), input(_input), idxptr(_idxptr) {}
        
        void _parse_value();
        size_t _parse_str(size_t idx); // Parse String from input[idx](") and return the index of parsed string in tape literals
        size_t _parse_array();
        size_t _parse_object();
    };
}

#endif