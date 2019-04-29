#include "tape.h"

#include <string.h>

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
                printf("\"%s\"", literals + (section >> 4U));
                return 1;
            case TYPE_INT:
                printf("%lld", static_cast<long long int>(tape[tape_idx + 1]));
                return 2;
            case TYPE_DEC:
                printf("%lf", plain_convert(static_cast<long long int>(tape[tape_idx + 1])));
                return 2;
            case TYPE_ARR: {
                size_t elem_idx = tape_idx + 1;
                bool first = true;
                printf("[");
                while (elem_idx < (section >> 4U)) {
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
                while (elem_idx < (section >> 4U)) {
                    if (first) first = false; else printf(",");
                    printf("\n");
                    print_indent(indent + 2);
                    printf("\"%s\": ", literals + (tape[elem_idx++] >> 4U));
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

#define __expect(__char) ({ \
        if (ch != (__char)) throw std::runtime_error("unexpected character"); \
    })

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
#else
        parse_str_naive(input, dest, &len, idx + 1);
#endif
        tape->literals_size += len + 1;
        return index;
    }
}
