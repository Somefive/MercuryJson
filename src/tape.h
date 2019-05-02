#ifndef MERCURYJSON_TAPE_H
#define MERCURYJSON_TAPE_H

#include <immintrin.h>
#include <stdio.h>
#include <atomic>

#include "mercuryparser.h"
#include "utils.h"


namespace MercuryJson {

    struct Tape {
        static const uint64_t TYPE_MASK = 0xf000000000000000;
        static const uint64_t TYPE_NULL = 0xf000000000000000;
        static const uint64_t TYPE_FALSE = 0x0000000000000000;
        static const uint64_t TYPE_TRUE = 0x1000000000000000;
        static const uint64_t TYPE_STR = 0x2000000000000000;
        static const uint64_t TYPE_INT = 0x3000000000000000;
        static const uint64_t TYPE_DEC = 0x4000000000000000;
        static const uint64_t TYPE_OBJ = 0x5000000000000000;
        static const uint64_t TYPE_ARR = 0x6000000000000000;

        uint64_t *tape;
        uint64_t *numeric;
        char *literals;
        size_t tape_size, literals_size, numeric_size;

        Tape(size_t string_size, size_t structural_size) {
            tape = static_cast<uint64_t *>(
                    aligned_malloc(2 * sizeof(uint64_t) * (structural_size + ALIGNMENT_SIZE)));
            numeric = static_cast<uint64_t *>(
                    aligned_malloc(sizeof(uint64_t) * (structural_size + ALIGNMENT_SIZE)));
            // literals = static_cast<char *>(aligned_malloc(string_size + ALIGNMENT_SIZE));
            tape_size = 0;
            literals_size = 0;
            numeric_size = 0;
        }

        ~Tape() {
            aligned_free(tape);
            // aligned_free(literals);
            aligned_free(numeric);
        }

        inline void write_null() { tape[tape_size++] = TYPE_NULL; }

        inline void write_true() { tape[tape_size++] = TYPE_TRUE; }

        inline void write_false() { tape[tape_size++] = TYPE_FALSE; }

        inline void write_integer(long long int value) {
            tape[tape_size++] = TYPE_INT | numeric_size;
            numeric[numeric_size++] = static_cast<uint64_t>(value);
        }

        inline void write_decimal(double value) {
            tape[tape_size++] = TYPE_DEC | numeric_size;
            numeric[numeric_size++] = static_cast<uint64_t>(plain_convert(value));
        }

        inline void write_str(uint64_t literal_idx) { tape[tape_size++] = TYPE_STR | literal_idx; }

        inline void write_array(size_t idx1, size_t idx2) {
            tape[idx1] = TYPE_ARR | idx2;
            tape[idx2] = TYPE_ARR | idx1;
        }

        inline void write_object(size_t idx1, size_t idx2) {
            tape[idx1] = TYPE_OBJ | idx2;
            tape[idx2] = TYPE_OBJ | idx1;
        }

        inline size_t write_array() {
            tape[tape_size] = TYPE_ARR;
            return tape_size++;
        }

        inline size_t write_object() {
            tape[tape_size] = TYPE_OBJ;
            return tape_size++;
        }

        inline void write_content(uint64_t content, size_t idx) { tape[idx] = (tape[idx] & TYPE_MASK) | content; }

        inline void append_content(uint64_t content, size_t idx) { tape[idx] |= content; }

        size_t print_json(size_t tape_idx = 0, size_t indent = 0);
        void print_tape();

        void state_machine(char *input, size_t *idxptr, size_t structural_size);
        void _parse_and_write_number(const char *input, size_t offset);
        void __parse_and_write_number(const char *input, size_t offset, size_t tape_idx, size_t numeric_idx);
        size_t _parse_str(char *input, size_t idx);
        std::atomic_bool reap;
        void _thread_parse_str(size_t pid, char *input, size_t *idxptr, size_t structural_size);
        void _thread_parse_num(size_t pid, char *input);
    };

    struct TapeWriter {
        Tape *tape;
        const char *input;
        size_t *idxptr;

        TapeWriter(Tape *_tape, const char *_input, size_t *_idxptr) : tape(_tape), input(_input), idxptr(_idxptr) {}

        void state_machine();
        void _parse_value();
        // parse string from input[idx](") and return the index of parsed string in tape literals
        size_t _parse_str(size_t idx);
        size_t _parse_array();
        size_t _parse_object();
    };
}

#endif // MERCURYJSON_TAPE_H
