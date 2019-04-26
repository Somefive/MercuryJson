#include <cstring>

#include "tape.h"
#include "mercuryparser.h"
#include "constants.h"
#include "utils.h"


namespace MercuryJson {

    size_t Tape::print_json(size_t tape_idx, size_t indent) {
        u_int64_t section = tape[tape_idx];
        switch (section & TYPE_MASK) {
            case TYPE_NULL: printf("null"); return 1;
            case TYPE_FALSE: printf("false"); return 1;
            case TYPE_TRUE: printf("true"); return 1;
            case TYPE_STR: printf("\"%s\"", literals + (section >> 4)); return 1;
            case TYPE_INT: printf("%lld", static_cast<long long int>(tape[tape_idx+1])); return 2;
            case TYPE_DEC: printf("%lf", plain_convert(static_cast<long long int>(tape[tape_idx+1]))); return 2;
            case TYPE_ARR: 
            {
                size_t elem_idx = tape_idx + 1; bool first = true;
                printf("[");
                while (elem_idx < (section >> 4)) {
                    if (first) first = false; else printf(",");
                    printf("\n"); print_indent(indent+2);
                    elem_idx += print_json(elem_idx, indent+2); 
                }
                printf("\n"); print_indent(indent); printf("]");
                return elem_idx + 1 - tape_idx;
            }
            case TYPE_OBJ: 
            {
                size_t elem_idx = tape_idx + 1; bool first = true;
                printf("{");
                while (elem_idx < (section >> 4)) {
                    if (first) first = false; else printf(",");
                    printf("\n"); print_indent(indent+2);
                    printf("\"%s\": ", literals + (tape[elem_idx++] >> 4));
                    elem_idx += print_json(elem_idx, indent+2);
                }
                printf("\n"); print_indent(indent); printf("}");
                return elem_idx + 1 - tape_idx;
            }
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
            case '-':
            {
                bool is_decimal;
                auto ret = parse_number(input, &is_decimal, idx);
                if (is_decimal) tape->write_decimal(std::get<double>(ret));
                else tape->write_integer(std::get<long long int>(ret));
                break;
            }
            case '[':
            {
                size_t left_tape_idx = tape->write_array();
                size_t right_tape_idx = _parse_array();
                tape->write_content(right_tape_idx, left_tape_idx);
                tape->write_content(left_tape_idx, right_tape_idx);
                break;
            }
            case '{':
            {
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
        if (ch == ']') { idxptr++; return tape->write_array(); }
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
            ch = input[*idxptr++];
        }
    }

    size_t TapeWriter::_parse_str(size_t idx) {
        char *src = input + idx + 1;
        size_t index = tape->literals_size;
        char *dest = tape->literals + index;
        char *base = dest;
        u_int64_t prev_odd_backslash_ending_mask = 0ULL;
        while (true) {
            Warp input(src);
            u_int64_t escape_mask = extract_escape_mask(input, &prev_odd_backslash_ending_mask);
            u_int64_t quote_mask = __cmpeq_mask<'"'>(input) & (~escape_mask);
            size_t ending_offset = _tzcnt_u64(quote_mask);

            if (ending_offset < 64) {
                size_t last_offset = 0, length;
                while (true) {
                    size_t this_offset = _tzcnt_u64(escape_mask);
                    if (this_offset >= ending_offset) {
                        memmove(dest, src + last_offset, ending_offset - last_offset);
                        dest += ending_offset - last_offset;
                        *dest++ = '\0';
                        break;
                    }
                    length = this_offset - last_offset;
                    char escaper = src[this_offset];
                    memmove(dest, src + last_offset, length);
                    dest += length;
                    *(dest - 1) = escape_map[escaper];
                    last_offset = this_offset + 1;
                    escape_mask = _blsr_u64(escape_mask);
                }
                break;
            } else {
                size_t last_offset = 0, length;
                while (true) {
                    size_t this_offset = _tzcnt_u64(escape_mask);
                    length = this_offset - last_offset;
                    char escaper = src[this_offset];
                    memmove(dest, src + last_offset, length);
                    dest += length;
                    if (this_offset >= ending_offset) break;
                    *(dest - 1) = escape_map[escaper];
                    last_offset = this_offset + 1;
                    escape_mask = _blsr_u64(escape_mask);
                }
                src += 64;
            }
        }
        tape->literals_size += dest - base;
        return index;
    }
}