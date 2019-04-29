#ifndef MERCURYJSON_TAPE_H
#define MERCURYJSON_TAPE_H

#include <immintrin.h>
#include <stdio.h>

#include "mercuryparser.h"
#include "utils.h"


namespace MercuryJson {

    struct Tape {
        static const uint64_t TYPE_MASK = 0xf;
        static const uint64_t TYPE_NULL = 0xf;
        static const uint64_t TYPE_FALSE = 0x0;
        static const uint64_t TYPE_TRUE = 0x1;
        static const uint64_t TYPE_STR = 0x2;
        static const uint64_t TYPE_INT = 0x3;
        static const uint64_t TYPE_DEC = 0x4;
        static const uint64_t TYPE_OBJ = 0x5;
        static const uint64_t TYPE_ARR = 0x6;

        uint64_t *tape;
        char *literals;
        size_t tape_size, literals_size;

        Tape(size_t string_size, size_t structual_size) {
            tape = static_cast<uint64_t *>(
                    aligned_malloc(2 * sizeof(uint64_t) * (structual_size + ALIGNMENT_SIZE)));
            literals = static_cast<char *>(aligned_malloc(string_size + ALIGNMENT_SIZE));
            tape_size = 0;
            literals_size = 0;
        }

        ~Tape() {
            delete[] tape;
            delete[] literals;
        }

        void write_null() { tape[tape_size++] = TYPE_NULL; }

        void write_true() { tape[tape_size++] = TYPE_TRUE; }

        void write_false() { tape[tape_size++] = TYPE_FALSE; }

        void write_integer(long long int value) {
            tape[tape_size++] = TYPE_INT;
            tape[tape_size++] = value;
        }

        void write_decimal(double value) {
            tape[tape_size++] = TYPE_DEC;
            tape[tape_size++] = plain_convert(value);
        }

        void write_str(uint64_t literal_idx) { tape[tape_size++] = TYPE_STR | (literal_idx << 4U); }

        size_t write_array() {
            tape[tape_size] = TYPE_ARR;
            return tape_size++;
        }

        size_t write_object() {
            tape[tape_size] = TYPE_OBJ;
            return tape_size++;
        }

        void write_content(uint64_t content, size_t idx) { tape[idx] = (tape[idx] & TYPE_MASK) | (content << 4U); }

        size_t print_json(size_t tape_idx = 0, size_t indent = 0);
    };

    struct TapeWriter {
        Tape *tape;
        char *input;
        size_t *idxptr;

        TapeWriter(Tape *_tape, char *_input, size_t *_idxptr) : tape(_tape), input(_input), idxptr(_idxptr) {}

        void _parse_value();
        // parse string from input[idx](") and return the index of parsed string in tape literals
        size_t _parse_str(size_t idx);
        size_t _parse_array();
        size_t _parse_object();
    };
}

#endif // MERCURYJSON_TAPE_H
