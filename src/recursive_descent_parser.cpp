#include "mercuryparser.h"


namespace MercuryJson {
#define next_char() ({                                                            \
        idx = *idx_ptr++;                                                         \
        if (idx >= input_len) throw std::runtime_error("text ended prematurely"); \
        ch = input[idx];                                                          \
    })

#define peek_char() ({   \
        idx = *idx_ptr;  \
        ch = input[idx]; \
    })

#define expect(__char) ({                             \
        if (ch != (__char)) _error(#__char, ch, idx); \
    })

#define error(__expected) ({         \
        _error(__expected, ch, idx); \
    })

    JsonValue *JSON::_parse_object() {
        size_t idx;
        char ch;
        peek_char();
        if (ch == '}') {
            next_char();
            return allocator.construct(static_cast<JsonObject *>(nullptr));
        }

        expect('"');
        char *str = _parse_str(idx);
        next_char();
        next_char();
        expect(':');
        JsonValue *value = _parse_value();
        auto *object = allocator.construct<JsonObject>(str, value), *ptr = object;
        while (true) {
            next_char();
            if (ch == '}') break;
            expect(',');
            peek_char();
            expect('"');
            str = _parse_str(idx);
            next_char();
            next_char();
            expect(':');
            value = _parse_value();
            ptr = ptr->next = allocator.construct<JsonObject>(str, value);
        }
        return allocator.construct(object);
    }

    JsonValue *JSON::_parse_array() {
        size_t idx;
        char ch;
        peek_char();
        if (ch == ']') {
            next_char();
            return allocator.construct(static_cast<JsonArray *>(nullptr));
        }
        JsonValue *value = _parse_value();
        auto *array = allocator.construct<JsonArray>(value), *ptr = array;
        while (true) {
            next_char();
            if (ch == ']') break;
            expect(',');
            value = _parse_value();
            ptr = ptr->next = allocator.construct<JsonArray>(value);
        }
        return allocator.construct(array);
    }

    JsonValue *JSON::_parse_value() {
        size_t idx;
        char ch;
        next_char();
        JsonValue *value;
        switch (ch) {
            case '"':
                value = allocator.construct(_parse_str(idx));
                break;
            case 't':
                value = allocator.construct(parse_true(input, idx));
                break;
            case 'f':
                value = allocator.construct(parse_false(input, idx));
                break;
            case 'n':
                parse_null(input, idx);
                value = allocator.construct();
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
                if (is_decimal) value = allocator.construct(std::get<double>(ret));
                else value = allocator.construct(std::get<long long int>(ret));
                break;
            }
            case '[':
                value = _parse_array();
                break;
            case '{':
                value = _parse_object();
                break;
            default:
                error("JSON value");
        }
        return value;
    }

#undef next_char
#undef peek_char
#undef expect
#undef error
}
