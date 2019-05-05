#include "tape.h"

#include <string.h>
#include <math.h>

#include <sstream>
#include <thread>
#include <future>

#include "constants.h"
#include "flags.h"
#include "mercuryparser.h"
#include "utils.h"


namespace MercuryJson {

    [[noreturn]] void _error(const char *expected, const char *input, char encountered, size_t index) {
        std::stringstream stream;
        char _encounter[3];
        size_t len = 0;
        __error_maybe_escape(_encounter, &len, encountered);
        _encounter[len] = 0;
        stream << "expected " << expected << " at index " << index << ", but encountered '" << _encounter << "'";
        MercuryJson::__error(stream.str(), input, index);
    }

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
                printf("\"%s\"", literals + (section & VALUE_MASK));
                return 1;
            case TYPE_INT:
                printf("%lld", static_cast<long long int>(numeric[section & VALUE_MASK]));
                return 1;
            case TYPE_DEC:
                printf("%.10lf", plain_convert(static_cast<long long int>(numeric[section & VALUE_MASK])));
                return 1;
            case TYPE_ARR: {
                size_t elem_idx = tape_idx + 1;
                bool first = true;
                printf("[");
                assert((section & VALUE_MASK) > tape_idx);
                while (elem_idx < (section & VALUE_MASK)) {
                    if (first) first = false; else printf(",");
                    printf("\n");
                    print_indent(indent + 2);
                    elem_idx += print_json(elem_idx, indent + 2);
                    if ((tape[elem_idx] & TYPE_MASK) == TYPE_JUMP) {
                        // Skip jumps at the end of each value.
                        // Otherwise, this case will fail:  [ value JUMP ]
                        elem_idx += tape[elem_idx] & VALUE_MASK;
                    }
                }
                assert(elem_idx == (section & VALUE_MASK) && tape_idx == (tape[elem_idx] & VALUE_MASK)
                       && (tape[elem_idx] & TYPE_MASK) == TYPE_ARR);
                printf("\n");
                print_indent(indent);
                printf("]");
                return elem_idx + 1 - tape_idx;
            }
            case TYPE_OBJ: {
                size_t elem_idx = tape_idx + 1;
                bool first = true;
                printf("{");
                assert((section & VALUE_MASK) > tape_idx);
                while (elem_idx < (section & VALUE_MASK)) {
                    if (first) first = false; else printf(",");
                    printf("\n");
                    print_indent(indent + 2);
                    elem_idx += print_json(elem_idx, indent + 2);
                    printf(": ");
                    elem_idx += print_json(elem_idx, indent + 2);
                    if ((tape[elem_idx] & TYPE_MASK) == TYPE_JUMP) {
                        // Skip jumps at the end of each value.
                        // Otherwise, this case will fail:  { str : value JUMP }
                        elem_idx += tape[elem_idx] & VALUE_MASK;
                    }
                }
                assert(elem_idx == (section & VALUE_MASK) && tape_idx == (tape[elem_idx] & VALUE_MASK)
                       && (tape[elem_idx] & TYPE_MASK) == TYPE_OBJ);
                printf("\n");
                print_indent(indent);
                printf("}");
                return elem_idx + 1 - tape_idx;
            }
            case TYPE_JUMP: {
                size_t offset = (section & VALUE_MASK);
                return print_json(tape_idx + offset, indent) + offset;
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
                    printf("string: \"%s\"\n", literals + (section & VALUE_MASK));
                    break;
                case TYPE_INT:
                    printf("integer: %lld\n", static_cast<long long int>(numeric[section & VALUE_MASK]));
                    break;
                case TYPE_DEC:
                    printf("decimal: %lf\n", plain_convert(static_cast<long long int>(numeric[section & VALUE_MASK])));
                    break;
                case TYPE_ARR:
                    printf("array: %llu\n", (section & VALUE_MASK));
                    break;
                case TYPE_OBJ:
                    printf("object: %llu\n", (section & VALUE_MASK));
                    break;
                case TYPE_JUMP:
                    printf("jump offset: %llu\n", (section & VALUE_MASK));
                    break;
                default:
                    printf("unknown: type = %llu, value = %llu\n", (section & TYPE_MASK), (section & VALUE_MASK));
//                    throw std::runtime_error("unexpected element on tape");
                    break;
            }
        }
    }

    static const size_t kMaxDepth = 1024;

    struct TapeStack {
        size_t depth, extra_closing_count;
        void *ret_address[kMaxDepth];
        size_t scope_offset[kMaxDepth];
        size_t extra_closing_offset[kMaxDepth];  // offsets of extra closing brackets

        TapeStack() : depth(0), extra_closing_count(0) {}

        inline void push(size_t offset, void *address) {
            scope_offset[depth] = offset;
            ret_address[depth] = address;
            ++depth;
        }
    };

#define peek_char() ({             \
        idx = indices[idx_offset]; \
        ch = input[idx];           \
    })

#define expect(__char) ({                                    \
        if (ch != (__char)) _error(#__char, input, ch, idx); \
    })

    void Tape::_thread_state_machine(char *input, const size_t *indices, size_t idx_begin, size_t idx_end,
                                     TapeStack *stack, size_t *tape_end, bool start_unknown) {

#define next_char() ({                           \
        if (idx_offset == idx_end) goto succeed; \
        idx = indices[idx_offset++];             \
        ch = input[idx];                         \
    })

        size_t idx;  // index in input
        char ch;  // current character
        size_t left_tape_idx, right_tape_idx;
        size_t idx_offset = idx_begin;
        // We keep a local counter for `tape_size` for the part of tape that the current thread writes to.
        // Our implementation only guarantees that tape size is no greater than the number of structural characters,
        // since commas (,) and colons (:) are not stored, and numerals and literals are stored off-tape.
        size_t tape_pos = idx_begin;

        if (start_unknown) {
            goto unknown_start;
        } else {
            goto start_value;
        }

#define PARSE_VALUE(continue_address) ({                                             \
            switch (ch) {                                                            \
                case '"':                                                            \
                    write_str(tape_pos++, _parse_str(input, idx));                   \
                    break;                                                           \
                case 't':                                                            \
                    parse_true(input, idx);                                          \
                    write_true(tape_pos++);                                          \
                    break;                                                           \
                case 'f':                                                            \
                    parse_false(input, idx);                                         \
                    write_false(tape_pos++);                                         \
                    break;                                                           \
                case 'n':                                                            \
                    parse_null(input, idx);                                          \
                    write_null(tape_pos++);                                          \
                    break;                                                           \
                case '0':                                                            \
                case '1':                                                            \
                case '2':                                                            \
                case '3':                                                            \
                case '4':                                                            \
                case '5':                                                            \
                case '6':                                                            \
                case '7':                                                            \
                case '8':                                                            \
                case '9':                                                            \
                case '-': {                                                          \
                    _parse_and_write_number(input, idx, tape_pos++, idx_offset - 1); \
                    break;                                                           \
                }                                                                    \
                case '[': {                                                          \
                    write_array(tape_pos);                                           \
                    stack->push(tape_pos++, continue_address);                       \
                    goto array_begin;                                                \
                }                                                                    \
                case '{': {                                                          \
                    write_object(tape_pos);                                          \
                    stack->push(tape_pos++, continue_address);                       \
                    goto object_begin;                                               \
                }                                                                    \
                default:                                                             \
                    goto fail;                                                       \
            }                                                                        \
        })

#ifdef DEBUG
# define __PRINT_INFO(s) ({ printf("%lu[%c]: %s\n", idx, ch, s); })
#else
# define __PRINT_INFO(s) ({})
#endif

start_value:
        __PRINT_INFO("parse value");
        next_char();
        PARSE_VALUE(&&start_continue);
        goto succeed;
start_continue:
        next_char();  // strip off the extra closing bracket at end
        goto succeed;

unknown_start:
        __PRINT_INFO("parse unknown");
        next_char();
        switch (ch) {
            case '"':
                write_str(tape_pos++, _parse_str(input, idx));
                peek_char();
                if (ch == ':') goto object_key_state;
                break;
            case ']':
                goto array_end;
            case '}':
                goto object_end;
            case ':':
                --idx_offset;
                goto object_key_state;
            case ',':
                goto unknown_2nd_value;
            default:
                PARSE_VALUE(&&unknown_continue);
        }
unknown_continue:
        __PRINT_INFO("parse unknown continue");
        next_char();
        switch (ch) {
            case ',':
                goto unknown_2nd_value;
            case ']':
                goto array_end;
            case '}':
                goto object_end;
            default:
                goto fail;
        }
unknown_2nd_value:
        __PRINT_INFO("parse unknown 2nd value");
        next_char();
        if (ch == '"') {
            write_str(tape_pos++, _parse_str(input, idx));
            peek_char();
            if (ch == ':') goto object_key_state;
            else goto array_continue;
        } else {
            goto array_value;
        }

object_begin:
        __PRINT_INFO("parse object");
        next_char();
        switch (ch) {
            case '"':
                write_str(tape_pos++, _parse_str(input, idx));
                goto object_key_state;
            case '}':
                goto object_end;
            default:
                goto fail;
        }
object_key_state:
        __PRINT_INFO("parse object key");
        next_char();
        expect(':');
        next_char();
        PARSE_VALUE(&&object_continue);
object_continue:
        __PRINT_INFO("parse object continue");
        next_char();
        switch (ch) {
            case ',':
                next_char();
                expect('"');
                write_str(tape_pos++, _parse_str(input, idx));
                goto object_key_state;
            case '}':
                goto object_end;
            default:
                goto fail;
        }
object_end:
        __PRINT_INFO("parse object end");
        if (stack->depth == 0) {
            // Extra closing curly bracket in current segment.
            stack->extra_closing_offset[stack->extra_closing_count++] = tape_pos;
            write_object(tape_pos);
            append_content(tape_pos, idx_offset - 1);
            ++tape_pos;
            goto unknown_continue;
        } else {
            --stack->depth;
            left_tape_idx = stack->scope_offset[stack->depth];
            right_tape_idx = tape_pos++;
            write_object(left_tape_idx, right_tape_idx);
            //@formatter:off
            goto *stack->ret_address[stack->depth];
            //@formatter:on
        }

array_begin:
        __PRINT_INFO("parse array");
        next_char();
        if (ch == ']') goto array_end;
array_value:
        __PRINT_INFO("parse array value");
        PARSE_VALUE(&&array_continue);
array_continue:
        __PRINT_INFO("parse array continue");
        next_char();
        switch (ch) {
            case ',':
                next_char();
                goto array_value;
            case ']':
                goto array_end;
            default:
                goto fail;
        }
array_end:
        __PRINT_INFO("parse array end");
        if (stack->depth == 0) {
            // Extra closing square bracket in current segment.
            stack->extra_closing_offset[stack->extra_closing_count++] = tape_pos;
            write_array(tape_pos);
            append_content(tape_pos, idx_offset - 1);
            ++tape_pos;
            goto unknown_continue;
        } else {
            --stack->depth;
            left_tape_idx = stack->scope_offset[stack->depth];
            right_tape_idx = tape_pos++;
            write_array(left_tape_idx, right_tape_idx);
            //@formatter:off
            goto *stack->ret_address[stack->depth];
            //@formatter:on
        }

fail:
        MercuryJson::__error("unexpected character when parsing value", input, idx);
succeed:
        if (idx_offset != idx_end) MercuryJson::__error("excessive characters at end of input", input, idx);
        __PRINT_INFO("parse succeed");
        *tape_end = tape_pos;

#undef next_char
#undef PARSE_VALUE
    }

    inline bool __is_num_or_str(char ch) {
        return ch == '"' || ch == '-' || (ch >= '0' && ch <= '9');
    }

    inline bool __is_opening_bracket(char ch) {
        return ch == '{' || ch == '[';
    }

    inline bool __is_closing_bracket(char ch) {
        return ch == '}' || ch == ']';
    }

    inline bool __is_separator(char ch) {
        return ch == ',' || ch == ':';
    }

    void Tape::state_machine(char *input, size_t *idx_ptr, size_t structural_size) {
#if PARSE_STR_NUM_THREADS
        std::thread parse_str_threads[PARSE_STR_NUM_THREADS];
        for (size_t i = 0; i < PARSE_STR_NUM_THREADS; ++i)
            parse_str_threads[i] = std::thread(&Tape::_thread_parse_str, this, i, input, idx_ptr, structural_size);
#endif
        literals = input;

#if TAPE_STATE_MACHINE_NUM_THREADS == 1
        TapeStack stack;
        _thread_state_machine(input, idx_ptr, 0, structural_size, &stack, &tape_size);
        if (stack.depth != 0) throw std::runtime_error("unclosed brackets at end of input");
#else
        TapeStack stack[TAPE_STATE_MACHINE_NUM_THREADS];
        std::future<void> parse_threads[TAPE_STATE_MACHINE_NUM_THREADS - 1];
        size_t tape_ends[TAPE_STATE_MACHINE_NUM_THREADS];
        size_t idx_splits[TAPE_STATE_MACHINE_NUM_THREADS + 1];
        for (int i = 0; i <= TAPE_STATE_MACHINE_NUM_THREADS; ++i)
            idx_splits[i] = (structural_size - 1) * i / TAPE_STATE_MACHINE_NUM_THREADS;
        for (int i = 1; i < TAPE_STATE_MACHINE_NUM_THREADS; ++i) {
            size_t idx_begin = idx_splits[i];
            size_t idx_end = idx_splits[i + 1];
            parse_threads[i - 1] = std::async(std::launch::async, &Tape::_thread_state_machine, this, input, idx_ptr,
                                              idx_begin, idx_end, &stack[i], &tape_ends[i], /*start_unknown=*/true);
//            parse_threads[i - 1].get();
        }
        _thread_state_machine(input, idx_ptr, idx_splits[0], idx_splits[1], &stack[0], &tape_ends[0]);

//        for (int pid = 1; pid < TAPE_STATE_MACHINE_NUM_THREADS; ++pid)
//            parse_threads[pid - 1].get();
//        for (int i = 0; i < TAPE_STATE_MACHINE_NUM_THREADS; ++i) {
//            size_t idx_begin = idx_splits[i];
//            printf("Segment #%d: tape span = [%lu, %lu), stack size = %lu, extra brackets = %lu\n",
//                   i, idx_begin, tape_ends[i], stack[i].depth, stack[i].extra_closing_count);
//            printf("  Extra: ");
//            for (int j = 0; j < stack[i].extra_closing_count; ++j) {
//                size_t offset = stack[i].extra_closing_offset[j];
//                size_t type = tape[offset] & TYPE_MASK;
//                printf(" %s(%lu)", type == TYPE_OBJ ? "obj" : "arr", offset);
//            }
//            printf("\n");
//            printf("  Stack: ");
//            for (int j = 0; j < stack[i].depth; ++j) {
//                size_t offset = stack[i].scope_offset[j];
//                size_t type = tape[offset] & TYPE_MASK;
//                printf(" %s(%lu)", type == TYPE_OBJ ? "obj" : "arr", offset);
//            }
//            printf("\n");
//        }

        // Join threads and merge.
        size_t merge_stack[kMaxDepth];
        size_t top = 0;
        for (int i = 0; i < stack[0].depth; ++i)
            merge_stack[top++] = stack[0].scope_offset[i];
        for (int pid = 1; pid < TAPE_STATE_MACHINE_NUM_THREADS; ++pid) {
            parse_threads[pid - 1].get();

            // Verify grammar correctness checks for cross-boundary input.
            // This is to make sure structural characters that are not stored on tape (, and :) are properly inserted
            // between segments, i.e. the following cases should fail when running with 2 threads:
            //  No.  Reason                    1st Thread         2nd Thread
            //   1.  Missing colon (:)         { "1": 2, "3"      4, "5": 6 }
            //   2.  Missing comma (,)         [ 1, 2             3, 4]
            //   3.  Missing comma (,)         [ [ 1, 2 ]         [ 3, 4 ] ]
            //   4.  Extra colon (:)           { "1": 2, "3":     : 4, "5": 6 }
            //   5.  Extra comma (,)           [ 1, 2,            , 3, 4 ]
            //   6.  Extra kv-pair in object   { "1": 2, "3":     "4": 5, "5": 6 }
            //   7.  Array value in object     { "1": 2,          "3", "5": 6 }
            size_t pos = idx_splits[pid];
            size_t idx = idx_ptr[pos];
            if (pos - 1 >= 0 && pos < structural_size) {
                char left_char = input[idx_ptr[pos - 1]], right_char = input[idx];
                if ((__is_num_or_str(left_char) && __is_num_or_str(right_char))  // cases 1 & 2
                    || (__is_closing_bracket(left_char) && __is_opening_bracket(right_char)))  // case 3
                    MercuryJson::__error("expected separator", input, idx);
                if (__is_separator(left_char) && __is_separator(right_char))  // cases 4 & 5
                    MercuryJson::__error("extra separator", input, idx);
            }
            bool in_object = (tape[merge_stack[top - 1]] & TYPE_MASK) == TYPE_OBJ;
            for (size_t right_pos = pos; right_pos <= pos + 1; ++right_pos) {
                if (right_pos - 2 >= 0 && right_pos < structural_size) {
                    char left_char = input[idx_ptr[right_pos - 2]], right_char = input[idx_ptr[right_pos]];
                    if (left_char == ':' && right_char == ':')  // case 6
                        MercuryJson::__error("extra colon (:)", input, idx_ptr[right_pos]);
                    if (in_object && left_char == ',' && right_char == ',')
                        MercuryJson::__error("non key-value pair in object", input, idx_ptr[right_pos]);
                }
            }

            TapeStack &cur_stack = stack[pid];
            for (int i = 0; i < cur_stack.extra_closing_count; ++i) {
                size_t right_tape_idx = cur_stack.extra_closing_offset[i];
                size_t right_input_idx = idx_ptr[tape[right_tape_idx] & VALUE_MASK];
                if (top == 0) MercuryJson::__error("unmatched closing bracket", input, right_input_idx);
                size_t left_tape_idx = merge_stack[--top];
                if ((tape[left_tape_idx] & TYPE_MASK) != (tape[right_tape_idx] & TYPE_MASK))
                    MercuryJson::__error("matching brackets have different types", input, right_input_idx);
                write_content(right_tape_idx, left_tape_idx);
                write_content(left_tape_idx, right_tape_idx);
            }
            for (int i = 0; i < cur_stack.depth; ++i)
                merge_stack[top++] = cur_stack.scope_offset[i];
        }
        if (top > 0) throw std::runtime_error("unmatched opening brackets");
        for (int i = 0; i < TAPE_STATE_MACHINE_NUM_THREADS - 1; ++i) {
            size_t idx_begin_next = idx_splits[i + 1];
            if (tape_ends[i] < idx_begin_next) write_jump(tape_ends[i], idx_begin_next);
        }
        tape_size = tape_ends[TAPE_STATE_MACHINE_NUM_THREADS - 1];

#endif

#if PARSE_NUM_NUM_THREADS && !NO_PARSE_NUMBER
        std::thread parse_num_threads[PARSE_STR_NUM_THREADS];
        for (size_t i = 0; i < PARSE_STR_NUM_THREADS; ++i)
            parse_num_threads[i] = std::thread(&Tape::_thread_parse_num, this, i, input, idx_ptr, structural_size);
        for (std::thread &thread : parse_num_threads)
            thread.join();
#endif
#if PARSE_STR_NUM_THREADS
        for (std::thread &thread : parse_str_threads)
            thread.join();
#endif
//        print_tape();
//        print_json();
//        printf("\n");
    }

    void Tape::__parse_and_write_number(const char *input, size_t offset, size_t tape_idx, size_t numeric_idx) {
        bool is_decimal;
        auto ret = parse_number(input, &is_decimal, offset);
        if (is_decimal) {
            tape[tape_idx] = TYPE_DEC | numeric_idx;
            numeric[numeric_idx] = *reinterpret_cast<uint64_t *>(&ret);
        } else {
            tape[tape_idx] = TYPE_INT | numeric_idx;
            numeric[numeric_idx] = *reinterpret_cast<uint64_t *>(&ret);
        }
    }

//    void Tape::__parse_and_write_number(const char *input, size_t offset, size_t tape_idx, size_t numeric_idx) {
//        const char *s = input + offset;
//        uint64_t integer = 0ULL;
//        bool negative = false;
//        int64_t exponent = 0LL;
//        if (*s == '-') {
//            ++s;
//            negative = true;
//        }
//        if (*s == '0') {
//            ++s;
//            if (*s >= '0' && *s <= '9')
//                throw std::runtime_error("numbers cannot have leading zeros");
//        } else {
//            while (*s >= '0' && *s <= '9')
//                integer = integer * 10 + (*s++ - '0');
//        }
//        if (*s == '.') {
//            const char *const base = ++s;
//#if PARSE_NUMBER_AVX
//            if (_all_digits(s)) {
//                integer += integer * 100000000 + _parse_eight_digits(s);
//                s += 8;
//            }
//#endif
//            while (*s >= '0' && *s <= '9')
//                integer = integer * 10 + (*s++ - '0');
//            exponent = base - s;
//        }
//        if (*s == 'e' || *s == 'E') {
//            ++s;
//            bool negative_exp = false;
//            if (*s == '-') {
//                negative_exp = true;
//                ++s;
//            } else if (*s == '+') ++s;
//            int64_t expo = 0LL;
//            while (*s >= '0' && *s <= '9')
//                expo = expo * 10 + (*s++ - '0');
//            exponent += negative_exp ? -expo : expo;
//        }
//        if (exponent == 0) {
//            tape[tape_idx] = TYPE_INT | numeric_idx;
//            numeric[numeric_idx] = negative ? -integer : integer;
//        } else {
//            if (exponent < -308 || exponent > 308) throw std::runtime_error("number out of range");
//            double decimal = negative ? -integer : integer;
//            decimal *= kPowerOfTen[308 + exponent];
//            tape[tape_idx] = TYPE_DEC | numeric_idx;
//            numeric[numeric_idx] = plain_convert(decimal);
//        }
//    }

    void Tape::_parse_and_write_number(const char *input, size_t offset, size_t tape_idx, size_t numeric_idx) {
#if NO_PARSE_NUMBER
        tape[tape_idx] = 0;
#elif PARSE_NUM_NUM_THREADS
        tape[tape_idx] = numeric_idx;
        numeric[numeric_idx] = tape_idx;
#else
        __parse_and_write_number(input, offset, tape_idx, numeric_idx);
#endif
    }

#define next_char() ({               \
        idx = indices[idx_offset++]; \
        ch = input[idx];             \
    })

    void TapeWriter::_parse_value() {

        size_t idx;
        char ch;
        next_char();
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
                long long int ret = parse_number(input, &is_decimal, idx);
                if (is_decimal) tape->write_decimal(plain_convert(ret));
                else tape->write_integer(ret);
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
        size_t idx;
        char ch;
        peek_char();
        if (ch == ']') {
            next_char();
            return tape->write_array();
        }
        while (true) {
            _parse_value();
            next_char();
            if (ch == ']') return tape->write_array();
            expect(',');
        }
    }

    size_t TapeWriter::_parse_object() {
        size_t idx;
        char ch;
        next_char();
        if (ch == '}') return tape->write_object();
        while (true) {
            expect('"');
            tape->write_str(_parse_str(idx));
            next_char();
            expect(':');
            _parse_value();
            next_char();
            if (ch == '}') return tape->write_object();
            expect(',');
            next_char();
        }
    }

#undef next_char
#undef peek_char
#undef expect

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

    void Tape::_thread_parse_str(size_t pid, char *input, const size_t *idx_ptr, size_t structural_size) {
#if PARSE_STR_NUM_THREADS
        size_t idx;
        size_t begin = pid * structural_size / PARSE_STR_NUM_THREADS;
        size_t end = (pid + 1) * structural_size / PARSE_STR_NUM_THREADS;
        if (end > structural_size) end = structural_size;
        for (size_t i = begin; i < end; ++i) {
            idx = idx_ptr[i];
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
#endif
    }

    void Tape::_thread_parse_num(size_t pid, char *input, const size_t *idx_ptr, size_t structural_size) {
#if PARSE_NUM_NUM_THREADS
        size_t begin = pid * structural_size / PARSE_NUM_NUM_THREADS;
        size_t end = (pid + 1) * structural_size / PARSE_NUM_NUM_THREADS;
        for (size_t numeric_idx = begin; numeric_idx < end; ++numeric_idx) {
            size_t offset = idx_ptr[numeric_idx];
            if (!(input[offset] == '-' || (input[offset] >= '0' && input[offset] <= '9'))) continue;
            size_t tape_idx = numeric[numeric_idx];
            __parse_and_write_number(input, offset, tape_idx, numeric_idx);
        }
#endif
    }
}
