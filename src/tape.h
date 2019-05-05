#ifndef MERCURYJSON_TAPE_H
#define MERCURYJSON_TAPE_H

#include <immintrin.h>
#include <stdio.h>
#include <atomic>

#include "mercuryparser.h"
#include "utils.h"


namespace MercuryJson {

    class Tape {
        static const uint64_t TYPE_MASK = 0xf000000000000000;
        static const uint64_t VALUE_MASK = ~TYPE_MASK;
        static const uint64_t TYPE_JUMP = 0x8000000000000000;  // skip empty tape positions when merging

        uint64_t *tape;
        // Numerals are also stored off-tape, in the `numeric` array, at the same offset as the structural character.
        // When using multi-threaded number parsing, during the main parsing algorithm, tape offsets for each number
        // are stored in `numeric`. This offset is then used in number parsing threads to write the number type.
        uint64_t *numeric;
        char *literals;
        size_t tape_size, literals_size, numeric_size;

        //@formatter:off
        inline void write_null() { write_null(tape_size++); }
        inline void write_null(size_t offset) {
            tape[offset] = TYPE_NULL;
        }

        inline void write_true() { write_true(tape_size++); }
        inline void write_true(size_t offset) {
            tape[offset] = TYPE_TRUE;
        }

        inline void write_false() { write_false(tape_size++); }
        inline void write_false(size_t offset) {
            tape[offset] = TYPE_FALSE;
        }

        inline void write_integer(long long int value) { write_integer(tape_size++, numeric_size++, value); }
        inline void write_integer(size_t offset, size_t num_offset, long long int value) {
            tape[offset] = TYPE_INT | numeric_size;
            numeric[num_offset] = static_cast<uint64_t>(value);
        }

        inline void write_decimal(double value) { write_decimal(tape_size++, numeric_size++, value); }
        inline void write_decimal(size_t offset, size_t num_offset, double value) {
            tape[offset] = TYPE_DEC | numeric_size;
            numeric[num_offset] = static_cast<uint64_t>(plain_convert(value));
        }

        inline void write_str(uint64_t literal_idx) { write_str(tape_size++, literal_idx); }
        inline void write_str(size_t offset, uint64_t literal_idx) {
            tape[offset] = TYPE_STR | literal_idx;
        }

        inline void write_array(size_t idx1, size_t idx2) {
            tape[idx1] = TYPE_ARR | idx2;
            tape[idx2] = TYPE_ARR | idx1;
        }

        inline void write_object(size_t idx1, size_t idx2) {
            tape[idx1] = TYPE_OBJ | idx2;
            tape[idx2] = TYPE_OBJ | idx1;
        }

        inline size_t write_array() {
            write_array(tape_size);
            return tape_size++;
        }
        inline void write_array(size_t offset) {
            tape[offset] = TYPE_ARR;
        }

        inline size_t write_object() {
            write_object(tape_size);
            return tape_size++;
        }
        inline void write_object(size_t offset) {
            tape[offset] = TYPE_OBJ;
        }
        //@formatter:on

        inline void write_jump(size_t offset, size_t dest) {
            tape[offset] = TYPE_JUMP | (dest - offset);
        }

        inline void write_content(size_t offset, uint64_t content) {
            tape[offset] = (tape[offset] & TYPE_MASK) | content;
        }

        inline void append_content(size_t offset, uint64_t content) {
            tape[offset] |= content;
        }

        inline void _parse_and_write_number(const char *input, size_t offset, size_t numeric_idx) {
            _parse_and_write_number(input, offset, tape_size++, numeric_idx);
        }

        void _parse_and_write_number(const char *input, size_t offset, size_t tape_idx, size_t numeric_idx);
        void __parse_and_write_number(const char *input, size_t offset, size_t tape_idx, size_t numeric_idx);
        size_t _parse_str(char *input, size_t idx);

        void _thread_parse_str(size_t pid, char *input, const size_t *idx_ptr, size_t structural_size);
        void _thread_parse_num(size_t pid, char *input, const size_t *idx_ptr, size_t structural_size);

        void _thread_state_machine(char *input, const size_t *indices, size_t idx_begin, size_t idx_end,
                                   struct TapeStack *stack, size_t *tape_end, bool start_unknown = false);

    public:
        static const uint64_t TYPE_NULL = 0xf000000000000000;
        static const uint64_t TYPE_FALSE = 0x1000000000000000;
        static const uint64_t TYPE_TRUE = 0x2000000000000000;
        static const uint64_t TYPE_STR = 0x3000000000000000;
        static const uint64_t TYPE_INT = 0x4000000000000000;
        static const uint64_t TYPE_DEC = 0x5000000000000000;
        static const uint64_t TYPE_OBJ = 0x6000000000000000;
        static const uint64_t TYPE_ARR = 0x7000000000000000;

        Tape(size_t string_size, size_t structural_size) {
            tape = aligned_malloc<uint64_t>(structural_size);
            numeric = aligned_malloc<uint64_t>(structural_size);
            memset(tape, 0, sizeof(uint64_t) * structural_size);
#if !TAPE_STATE_MACHINE
            literals = static_cast<char *>(aligned_malloc(string_size + kAlignmentSize));
#endif
            tape_size = 0;
            literals_size = 0;
            numeric_size = 0;
        }

        ~Tape() {
            aligned_free(tape);
#if !TAPE_STATE_MACHINE
            aligned_free(literals);
#endif
            aligned_free(numeric);
        }

        friend class TapeWriter;

        size_t print_json(size_t tape_idx = 0, size_t indent = 0);
        void print_tape();

        void state_machine(char *input, size_t *idx_ptr, size_t structural_size);
    };

    class TapeWriter {
        Tape *tape;
        const char *input;
        const size_t *indices;
        size_t idx_offset;

        void _parse_value();
        // parse string from input[idx](") and return the index of parsed string in tape literals
        size_t _parse_str(size_t idx);
        size_t _parse_array();
        size_t _parse_object();

    public:
        TapeWriter(Tape *tape, const char *input, size_t *indices) : tape(tape), input(input), indices(indices) {}

        inline void parse_value() {
            _parse_value();
        }
    };
}

#endif // MERCURYJSON_TAPE_H
