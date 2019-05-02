#include "tape.h"

#include <string.h>
#include <math.h>
#include <thread>

#include "constants.h"
#include "flags.h"
#include "mercuryparser.h"
#include "utils.h"


namespace MercuryJson {

    size_t Tape::print_json(size_t tape_idx, size_t indent) {
        uint64_t section = tape[tape_idx];
        switch (section & TYPE_MASK) {
            case TYPE_NULL:
                printf("null");
                return 1;
            case TYPE_FALSE:
                printf("false");
                return 1;
            case TYPE_TRUE:
                printf("true");
                return 1;
            case TYPE_STR:
                printf("\"%s\"", literals + (section & ~TYPE_MASK));
                return 1;
            case TYPE_INT:
                printf("%lld", static_cast<long long int>(numeric[section & ~TYPE_MASK]));
                return 1;
            case TYPE_DEC:
                printf("%lf", plain_convert(static_cast<long long int>(numeric[section & ~TYPE_MASK])));
                return 1;
            case TYPE_ARR: {
                size_t elem_idx = tape_idx + 1;
                bool first = true;
                printf("[");
                while (elem_idx < (section & ~TYPE_MASK)) {
                    if (first) first = false; else printf(",");
                    printf("\n");
                    print_indent(indent + 2);
                    elem_idx += print_json(elem_idx, indent + 2);
                }
                printf("\n");
                print_indent(indent);
                printf("]");
                return elem_idx + 1 - tape_idx;
            }
            case TYPE_OBJ: {
                size_t elem_idx = tape_idx + 1;
                bool first = true;
                printf("{");
                while (elem_idx < (section & ~TYPE_MASK)) {
                    if (first) first = false; else printf(",");
                    printf("\n");
                    print_indent(indent + 2);
                    printf("\"%s\": ", literals + (tape[elem_idx++] & ~TYPE_MASK));
                    elem_idx += print_json(elem_idx, indent + 2);
                }
                printf("\n");
                print_indent(indent);
                printf("}");
                return elem_idx + 1 - tape_idx;
            }
            default:
                throw std::runtime_error("unexpected element on tape");
        }
    }

    void Tape::print_tape() {
        for (size_t i = 0; i < tape_size; ++i) {
            printf("[%3lu] ", i);
            uint64_t section = tape[i];
            switch (section & TYPE_MASK) {
                case TYPE_NULL:
                    printf("null\n");
                    break;
                case TYPE_FALSE:
                    printf("false\n");
                    break;
                case TYPE_TRUE:
                    printf("true\n");
                    break;
                case TYPE_STR:
                    printf("string: \"%s\"\n", literals + (section & ~TYPE_MASK));
                    break;
                case TYPE_INT:
                    printf("integer: %lld\n", static_cast<long long int>(numeric[section & ~TYPE_MASK]));
                    break;
                case TYPE_DEC:
                    printf("decimal: %lf\n", plain_convert(static_cast<long long int>(numeric[section & ~TYPE_MASK])));
                    break;
                case TYPE_ARR:
                    printf("array: %llu\n", (section & ~TYPE_MASK));
                    break;
                case TYPE_OBJ:
                    printf("object: %llu\n", (section & ~TYPE_MASK));
                    break;
                default:
                    printf("unknown: %llu, %llu\n", (section & TYPE_MASK), (section & ~TYPE_MASK));
                    throw std::runtime_error("unexpected element on tape");
                    break;
            }
        }
    }

#define __expect(__char) ({ \
        if (ch != (__char)) throw std::runtime_error("unexpected character"); \
    })


#define MAX_DEPTH 1024

    void Tape::state_machine(char *input, size_t *idxptr, size_t structural_size) {
#if PARSE_STR_NUM_THREADS
        std::thread parse_str_threads[PARSE_STR_NUM_THREADS];
        for (size_t i = 0; i < PARSE_STR_NUM_THREADS; ++i)
            parse_str_threads[i] = std::thread(&Tape::_thread_parse_str, this, i, input, idxptr, structural_size);
#endif


        literals = input;

        void *ret_address[MAX_DEPTH];
        size_t scope_offset[MAX_DEPTH];

        size_t i = 0; // index in structural characters
        size_t idx; // index in input
        char ch; // current character
        size_t depth = 0;
        ret_address[depth++] = &&succeed;
        size_t left_tape_idx, right_tape_idx;

#define NEXT() ({ idx = idxptr[i++]; ch = input[idx]; })
#define PARSE_VALUE(continue_address) ({                                        \
            switch (ch) {                                                       \
                case '"':                                                       \
                    write_str(_parse_str(input, idx));                          \
                    break;                                                      \
                case 't':                                                       \
                    parse_true(input, idx);                                     \
                    write_true();                                               \
                    break;                                                      \
                case 'f':                                                       \
                    parse_false(input, idx);                                    \
                    write_false();                                              \
                    break;                                                      \
                case 'n':                                                       \
                    parse_null(input, idx);                                     \
                    write_null();                                               \
                    break;                                                      \
                case '0':                                                       \
                case '1':                                                       \
                case '2':                                                       \
                case '3':                                                       \
                case '4':                                                       \
                case '5':                                                       \
                case '6':                                                       \
                case '7':                                                       \
                case '8':                                                       \
                case '9':                                                       \
                case '-': {                                                     \
                    _parse_and_write_number(input, idx);                        \
                    break;                                                      \
                }                                                               \
                case '[': {                                                     \
                    scope_offset[depth] = tape_size++;                          \
                    ret_address[depth] = continue_address;                      \
                    depth++;                                                    \
                    goto array_begin;                                           \
                }                                                               \
                case '{': {                                                     \
                    scope_offset[depth] = tape_size++;                          \
                    ret_address[depth] = continue_address;                      \
                    depth++;                                                    \
                    goto object_begin;                                          \
                }                                                               \
                default:                                                        \
                    goto fail;                                                  \
            }                                                                   \
        })

#ifdef DEBUG
# define __PRINT_INFO(s) ({ printf("%lu[%c]: %s\n", idx, ch, s); })
#else
# define __PRINT_INFO(s) ({})
#endif

        NEXT();
value_parse:
        __PRINT_INFO("parse value");
        PARSE_VALUE(&&start_continue);
start_continue:
        goto succeed;

object_begin:
        __PRINT_INFO("parse object");
        NEXT();
        switch (ch) {
            case '"':
                write_str(_parse_str(input, idx));
                goto object_key_state;
            case '}':
                goto object_end;
            default:
                goto fail;
        }
object_key_state:
        __PRINT_INFO("parse object key");
        NEXT();
        __expect(':');
        NEXT();
        PARSE_VALUE(&&object_continue);
object_continue:
        __PRINT_INFO("parse object continue");
        NEXT();
        switch (ch) {
            case ',':
                NEXT();
                __expect('"');
                write_str(_parse_str(input, idx));
                goto object_key_state;
            case '}':
                goto object_end;
            default:
                goto fail;
        }
object_end:
        __PRINT_INFO("parse object end");
        --depth;
        left_tape_idx = scope_offset[depth];
        right_tape_idx = tape_size++;
        write_object(left_tape_idx, right_tape_idx);
        goto *ret_address[depth];

array_begin:
        __PRINT_INFO("parse array");
        NEXT();
        if (ch == ']') goto array_end;
array_value:
        __PRINT_INFO("parse array value");
        PARSE_VALUE(&&array_continue);
array_continue:
        __PRINT_INFO("parse array continue");
        NEXT();
        switch (ch) {
            case ',':
                NEXT();
                goto array_value;
            case ']':
                goto array_end;
            default:
                goto fail;
        }
array_end:
        __PRINT_INFO("parse array end");
        --depth;
        left_tape_idx = scope_offset[depth];
        right_tape_idx = tape_size++;
        write_array(left_tape_idx, right_tape_idx);
        goto *ret_address[depth];
fail:
        throw std::runtime_error("unexpected character when parsing value");
succeed:
        __PRINT_INFO("parse succeed");
        if (--depth != 0) goto fail;
#if PARSE_NUM_NUM_THREADS && !NO_PARSE_NUMBER
        std::thread parse_num_threads[PARSE_STR_NUM_THREADS];
        for (size_t i = 0; i < PARSE_STR_NUM_THREADS; ++i)
            parse_num_threads[i] = std::thread(&Tape::_thread_parse_num, this, i, input);
        for (std::thread &thread : parse_num_threads)
            thread.join();
#endif
#if PARSE_STR_NUM_THREADS
        for (std::thread &thread : parse_str_threads)
            thread.join();
#endif
        return;
#undef NEXT
#undef PARSE_VALUE
    }

    void Tape::__parse_and_write_number(const char *input, size_t offset, size_t tape_idx, size_t numeric_idx) {
        const char *s = input + offset;
        uint64_t integer = 0ULL;
        bool negative = false;
        int64_t exponent = 0LL;
        if (*s == '-') {
            ++s;
            negative = true;
        }
        if (*s == '0') {
            ++s;
            if (*s >= '0' && *s <= '9')
                throw std::runtime_error("numbers cannot have leading zeros");
        } else {
            while (*s >= '0' && *s <= '9')
                integer = integer * 10 + (*s++ - '0');
        }
        if (*s == '.') {
            const char *const base = ++s;
#if PARSE_NUMBER_AVX
            if (_all_digits(s)) {
                integer += integer * 100000000 + _parse_eight_digits(s);
                s += 8;
            }
#endif
            while (*s >= '0' && *s <= '9')
                integer = integer * 10 + (*s++ - '0');
            exponent = base - s;
        }
        if (*s == 'e' || *s == 'E') {
            ++s;
            bool negative_exp = false;
            if (*s == '-') {
                negative_exp = true;
                ++s;
            } else if (*s == '+') ++s;
            int64_t expo = 0LL;
            while (*s >= '0' && *s <= '9')
                expo = expo * 10 + (*s++ - '0');
            exponent += negative_exp ? -expo : expo;
        }
        if (exponent == 0) {
            tape[tape_idx] = TYPE_INT | numeric_idx;
            numeric[numeric_idx] = negative ? -integer : integer;
        } else {
            if (exponent < -308 || exponent > 308) throw new std::runtime_error("number out of range");
            double decimal = negative ? -integer : integer;
            decimal *= kPowerOfTen[308 + exponent];
            tape[tape_idx] = TYPE_DEC | numeric_idx;
            numeric[numeric_idx] = plain_convert(decimal);
        }
    }

    void Tape::_parse_and_write_number(const char *input, size_t offset) {
#if NO_PARSE_NUMBER
        tape[tape_size++] = 0;
        return;
#endif
#if PARSE_NUM_NUM_THREADS
        tape[tape_size] = offset;
        numeric[numeric_size] = tape_size;
        tape_size++;
        numeric_size++;
        return;
#endif
        __parse_and_write_number(input, offset, tape_size, numeric_size);
        tape_size++;
        numeric_size++;
    }


    void TapeWriter::_parse_value() {
        size_t idx = *idxptr++;
        char ch = input[idx];
        switch (ch) {
            case '"':
                tape->write_str(_parse_str(idx));
                break;
            case 't':
                parse_true(input, idx);
                tape->write_true();
                break;
            case 'f':
                parse_false(input, idx);
                tape->write_false();
                break;
            case 'n':
                parse_null(input, idx);
                tape->write_null();
                break;
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            case '-': {
                bool is_decimal;
                auto ret = parse_number(input, &is_decimal, idx);
                if (is_decimal) tape->write_decimal(std::get<double>(ret));
                else tape->write_integer(std::get<long long int>(ret));
                break;
            }
            case '[': {
                size_t left_tape_idx = tape->write_array();
                size_t right_tape_idx = _parse_array();
                tape->write_content(right_tape_idx, left_tape_idx);
                tape->write_content(left_tape_idx, right_tape_idx);
                break;
            }
            case '{': {
                size_t left_tape_idx = tape->write_object();
                size_t right_tape_idx = _parse_object();
                tape->write_content(right_tape_idx, left_tape_idx);
                tape->write_content(left_tape_idx, right_tape_idx);
                break;
            }
            default:
                throw std::runtime_error("unexpected character when parsing value");
        }
    }

    size_t TapeWriter::_parse_array() {
        char ch = input[*idxptr];
        if (ch == ']') {
            idxptr++;
            return tape->write_array();
        }
        while (true) {
            _parse_value();
            ch = input[*idxptr++];
            if (ch == ']') return tape->write_array();
            __expect(',');
        }
    }

    size_t TapeWriter::_parse_object() {
        size_t idx = *idxptr++;
        char ch = input[idx];
        if (ch == '}') return tape->write_object();
        while (true) {
            __expect('"');
            tape->write_str(_parse_str(idx));
            ch = input[*idxptr++];
            __expect(':');
            _parse_value();
            ch = input[*idxptr++];
            if (ch == '}') return tape->write_object();
            __expect(',');
            ch = input[idx = *idxptr++];
        }
    }

#undef __expect

    size_t TapeWriter::_parse_str(size_t idx) {
        size_t index = tape->literals_size;
        char *dest = tape->literals + index;
        size_t len;
#if PARSE_STR_MODE == 2
        parse_str_per_bit(input, dest, &len, idx + 1);
#elif PARSE_STR_MODE == 1
        parse_str_avx(input, dest, &len, idx + 1);
#elif PARSE_STR_MODE == 0
        parse_str_naive(input, dest, &len, idx + 1);
#else
        len = 0;
#endif
        tape->literals_size += len + 1;
        return index;
    }

    size_t Tape::_parse_str(char *input, size_t idx) {
#if PARSE_STR_NUM_THREADS
        return idx + 1;
#endif
        // size_t index = literals_size;
        // char *dest = literals + index;
        size_t index = idx;
        char *dest = input + idx + 1;
        size_t len;
#if PARSE_STR_MODE == 2
        parse_str_per_bit(input, dest, &len, idx + 1);
#elif PARSE_STR_MODE == 1
        parse_str_avx(input, dest, &len, idx + 1);
#elif PARSE_STR_MODE == 0
        parse_str_naive(input, dest, &len, idx + 1);
#else
        len = 0;
#endif
        literals_size += len + 1;
        return index + 1;
    }

    void Tape::_thread_parse_str(size_t pid, char *input, size_t *idxptr, size_t structural_size) {
#if PARSE_STR_NUM_THREADS
        size_t idx;
        size_t begin = pid * structural_size / PARSE_STR_NUM_THREADS;
        size_t end = (pid + 1) * structural_size / PARSE_STR_NUM_THREADS;
        if (end > structural_size) end = structural_size;
        for (size_t i = begin; i < end; ++i) {
            idx = idxptr[i];
            char *dest = input + idx + 1;
            if (input[idx] == '"') {
# if PARSE_STR_MODE == 2
                parse_str_per_bit(input, dest, nullptr, idx + 1);
# elif PARSE_STR_MODE == 1
                parse_str_avx(input, dest, nullptr, idx + 1);
# elif PARSE_STR_MODE == 0
                parse_str_naive(input, dest, nullptr, idx + 1);
# endif
            }
        }
# endif
    }

    void Tape::_thread_parse_num(size_t pid, char *input) {
# if PARSE_NUM_NUM_THREADS
        size_t begin = pid * numeric_size / PARSE_NUM_NUM_THREADS;
        size_t end = (pid + 1) * numeric_size / PARSE_NUM_NUM_THREADS;
        if (end > numeric_size) end = numeric_size;
        for (size_t numeric_idx = begin; numeric_idx < end; ++numeric_idx) {
            size_t tape_idx = numeric[numeric_idx];
            size_t offset = tape[tape_idx];
            __parse_and_write_number(input, offset, tape_idx, numeric_idx);
        }
# endif
    }
}
