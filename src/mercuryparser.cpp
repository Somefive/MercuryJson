#include "mercuryparser.h"

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <bitset>
#include <deque>
#include <map>
#include <sstream>
#include <stack>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "constants.h"


namespace MercuryJson {

    void __printChar(Warp &raw) {
        auto *vals = reinterpret_cast<uint8_t *>(&raw);
        for (size_t i = 0; i < 64; ++i) printf("%2x(%c) ", vals[i], vals[i]);
        printf("\n");
    }

    static const uint64_t kEvenMask64 = 0x5555555555555555U;
    static const uint64_t kOddMask64 = ~kEvenMask64;

    // @formatter:off
    uint64_t extract_escape_mask(const Warp &raw, uint64_t *prev_odd_backslash_ending_mask) {
        uint64_t backslash_mask = __cmpeq_mask<'\\'>(raw);
        uint64_t start_backslash_mask = backslash_mask & ~(backslash_mask << 1U);
        uint64_t even_start_backslash_mask = (start_backslash_mask & kEvenMask64) ^ *prev_odd_backslash_ending_mask;
        uint64_t even_carrier_backslash_mask = even_start_backslash_mask + backslash_mask;
        uint64_t even_escape_mask;
        even_escape_mask = (even_carrier_backslash_mask ^ backslash_mask) & kOddMask64;

        uint64_t odd_start_backslash_mask = (start_backslash_mask & kOddMask64) ^ *prev_odd_backslash_ending_mask;
        uint64_t odd_carrier_backslash_mask = odd_start_backslash_mask + backslash_mask;
        uint64_t odd_backslash_ending_mask = odd_carrier_backslash_mask < odd_start_backslash_mask;
        *prev_odd_backslash_ending_mask = odd_backslash_ending_mask;
        uint64_t odd_escape_mask;
        odd_escape_mask = (odd_carrier_backslash_mask ^ backslash_mask) & kEvenMask64;
        return even_escape_mask | odd_escape_mask;
    }
    // @formatter:on

    uint64_t extract_literal_mask(
            const Warp &raw, uint64_t escape_mask, uint64_t *prev_literal_ending, uint64_t *quote_mask) {
        *quote_mask = __cmpeq_mask<'"'>(raw) & ~escape_mask;
        uint64_t literal_mask = _mm_cvtsi128_si64(
                _mm_clmulepi64_si128(_mm_set_epi64x(0ULL, *quote_mask), _mm_set1_epi8(0xFF), 0));
        uint64_t literal_reversor = *prev_literal_ending * ~0ULL;
        literal_mask ^= literal_reversor;
        *prev_literal_ending = literal_mask >> 63U;
        return literal_mask;
    }

    void extract_structural_whitespace_characters(
            const Warp &raw, uint64_t literal_mask, uint64_t *structural_mask, uint64_t *whitespace_mask) {
        const __m256i upper_lookup = _mm256_setr_epi8(8, 0, 17, 2, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 17, 2, 0,
                                                      4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0);
        const __m256i lower_lookup = _mm256_setr_epi8(16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 4, 1, 12, 0, 0, 16, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 8, 10, 4, 1, 12, 0, 0);
        __m256i hi_upper_index = _mm256_shuffle_epi8(upper_lookup, _mm256_and_si256(_mm256_srli_epi16(raw.hi, 4),
                                                                                    _mm256_set1_epi8(0x7F)));
        __m256i hi_lower_index = _mm256_shuffle_epi8(lower_lookup, raw.hi);
        __m256i hi_character_label = _mm256_and_si256(hi_upper_index, hi_lower_index);
        __m256i lo_upper_index = _mm256_shuffle_epi8(upper_lookup, _mm256_and_si256(_mm256_srli_epi16(raw.lo, 4),
                                                                                    _mm256_set1_epi8(0x7F)));
        __m256i lo_lower_index = _mm256_shuffle_epi8(lower_lookup, raw.lo);
        __m256i lo_character_label = _mm256_and_si256(lo_upper_index, lo_lower_index);

        __m256i hi_whitespace_mask = _mm256_and_si256(hi_character_label, _mm256_set1_epi8(0x18));
        __m256i lo_whitespace_mask = _mm256_and_si256(lo_character_label, _mm256_set1_epi8(0x18));
        *whitespace_mask = ~__cmpeq_mask<0>(Warp(hi_whitespace_mask, lo_whitespace_mask)) & (~literal_mask);

        __m256i hi_structural_mask = _mm256_and_si256(hi_character_label, _mm256_set1_epi8(0x7));
        __m256i lo_structural_mask = _mm256_and_si256(lo_character_label, _mm256_set1_epi8(0x7));
        *structural_mask = ~__cmpeq_mask<0>(Warp(hi_structural_mask, lo_structural_mask)) & (~literal_mask);
    }

    uint64_t extract_pseudo_structural_mask(
            uint64_t structural_mask, uint64_t whitespace_mask, uint64_t quote_mask, uint64_t literal_mask,
            uint64_t *prev_pseudo_structural_end_mask) {
        uint64_t st_ws = structural_mask | whitespace_mask;
        structural_mask |= quote_mask;
        uint64_t pseudo_structural_mask = (st_ws << 1U) | *prev_pseudo_structural_end_mask;
        *prev_pseudo_structural_end_mask = (st_ws >> 63U) & 1ULL;
        pseudo_structural_mask &= (~whitespace_mask) & (~literal_mask);
        structural_mask |= pseudo_structural_mask;
        structural_mask &= ~(quote_mask & ~literal_mask);
        return structural_mask;
    }

    void construct_structural_character_pointers(
            uint64_t pseudo_structural_mask, size_t offset, size_t *indices, size_t *base) {
        size_t next_base = *base + __builtin_popcountll(pseudo_structural_mask);
        while (pseudo_structural_mask) {
            indices[*base] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[*base + 1] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[*base + 2] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[*base + 3] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[*base + 4] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[*base + 5] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[*base + 6] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[*base + 7] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            *base += 8;
        }
        *base = next_base;
    }

    void __printChar_m256i(__m256i raw) {
        auto *vals = reinterpret_cast<uint8_t *>(&raw);
        for (size_t i = 0; i < 32; ++i) printf("%2x(%c) ", vals[i], vals[i]);
        printf("\n");
    }

    // @formatter:off
    __mmask32 extract_escape_mask(__m256i raw, __mmask32 *prev_odd_backslash_ending_mask) {
        __mmask32 backslash_mask = __cmpeq_mask<'\\'>(raw);
        __mmask32 start_backslash_mask = backslash_mask & ~(backslash_mask << 1U);
        __mmask32 even_start_backslash_mask = (start_backslash_mask & __even_mask) ^ *prev_odd_backslash_ending_mask;
        __mmask32 even_carrier_backslash_mask = even_start_backslash_mask + backslash_mask;
        __mmask32 even_escape_mask = even_carrier_backslash_mask & (~backslash_mask) & __odd_mask;

        __mmask32 odd_start_backslash_mask = (start_backslash_mask & __odd_mask) ^ *prev_odd_backslash_ending_mask;
        __mmask32 odd_carrier_backslash_mask = odd_start_backslash_mask + backslash_mask;
        __mmask32 odd_backslash_ending_mask = odd_carrier_backslash_mask < odd_start_backslash_mask;
        *prev_odd_backslash_ending_mask = odd_backslash_ending_mask;
        __mmask32 odd_escape_mask = odd_carrier_backslash_mask & (~backslash_mask) & __even_mask;

        return even_escape_mask | odd_escape_mask;
    }
    // @formatter:on

    __mmask32 extract_literal_mask(
            __m256i raw, __mmask32 escape_mask, __mmask32 *prev_literal_ending, __mmask32 *quote_mask) {
        *quote_mask = __cmpeq_mask<'"'>(raw) & ~escape_mask;
        __mmask32 literal_mask = _mm_cvtsi128_si32(
                _mm_clmulepi64_si128(_mm_set_epi32(0, 0, 0, *quote_mask), _mm_set1_epi8(0xFF), 0));
        __mmask32 literal_reversor = *prev_literal_ending * ~0U;
        literal_mask ^= literal_reversor;
        *prev_literal_ending = literal_mask >> 31U;
        return literal_mask;
    }

    void extract_structural_whitespace_characters(
            __m256i raw, __mmask32 literal_mask, __mmask32 *structural_mask, __mmask32 *whitespace_mask) {
        const __m256i hi_lookup = _mm256_setr_epi8(8, 0, 17, 2, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 17, 2, 0, 4,
                                                   0, 4, 0, 0, 0, 0, 0, 0, 0, 0);
        const __m256i lo_lookup = _mm256_setr_epi8(16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 4, 1, 12, 0, 0, 16, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 8, 10, 4, 1, 12, 0, 0);
        __m256i hi_index = _mm256_shuffle_epi8(hi_lookup,
                                               _mm256_and_si256(_mm256_srli_epi16(raw, 4), _mm256_set1_epi8(0x7F)));
        __m256i lo_index = _mm256_shuffle_epi8(lo_lookup, raw);
        __m256i character_label = _mm256_and_si256(hi_index, lo_index);
        *whitespace_mask = ~__cmpeq_mask<0>(_mm256_and_si256(character_label, _mm256_set1_epi8(0x18))) & ~literal_mask;
        *structural_mask = ~__cmpeq_mask<0>(_mm256_and_si256(character_label, _mm256_set1_epi8(0x7))) & ~literal_mask;
    }

    __mmask32 extract_pseudo_structural_mask(
            __mmask32 structural_mask, __mmask32 whitespace_mask, __mmask32 quote_mask, __mmask32 literal_mask,
            __mmask32 *prev_pseudo_structural_end_mask) {
        __mmask32 st_ws = structural_mask | whitespace_mask;
        structural_mask |= quote_mask;
        __mmask32 pseudo_structural_mask = (st_ws << 1U) | *prev_pseudo_structural_end_mask;
        *prev_pseudo_structural_end_mask = st_ws >> 31U;
        pseudo_structural_mask &= (~whitespace_mask) & (~literal_mask);
        structural_mask |= pseudo_structural_mask;
        structural_mask &= ~(quote_mask & ~literal_mask);
        return structural_mask;
    }

    const size_t STRUCTURAL_UNROLL_COUNT = 8;

    void construct_structural_character_pointers(
            __mmask32 pseudo_structural_mask, size_t offset, size_t *indices, size_t *base) {
        size_t next_base = *base + __builtin_popcount(pseudo_structural_mask);
        while (pseudo_structural_mask) {
            // let the compiler unroll the loop
            for (size_t i = 0; i < STRUCTURAL_UNROLL_COUNT; ++i) {
                indices[*base + i] = offset + _tzcnt_u32(pseudo_structural_mask);
                pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            }
            *base += STRUCTURAL_UNROLL_COUNT;
        }
        *base = next_base;
    }

    void print_json(JsonValue *value, int indent) {
        int cnt;
        switch (value->type) {
            case JsonValue::TYPE_NULL:
                std::cout << "null";
                break;
            case JsonValue::TYPE_BOOL:
                if (value->boolean) std::cout << "true";
                else std::cout << "false";
                break;
            case JsonValue::TYPE_STR:
                std::cout << "\"" << value->str << "\"";
                break;
            case JsonValue::TYPE_OBJ:
                std::cout << "{" << std::endl;
                cnt = 0;
                for (auto *elem = value->object; elem; elem = elem->next) {
                    print_indent(indent + 2);
                    std::cout << "\"" << elem->key << "\": ";
                    print_json(elem->value, indent + 2);
                    if (elem->next != nullptr)
                        std::cout << ",";
                    std::cout << std::endl;
                    ++cnt;
                }
                print_indent(indent);
                std::cout << "}";
                break;
            case JsonValue::TYPE_ARR:
                std::cout << "[" << std::endl;
                cnt = 0;
                for (auto *elem = value->array; elem; elem = elem->next) {
                    print_indent(indent + 2);
                    print_json(elem->value, indent + 2);
                    if (elem->next != nullptr)
                        std::cout << ",";
                    std::cout << std::endl;
                    ++cnt;
                }
                print_indent(indent);
                std::cout << "]";
                break;
            case JsonValue::TYPE_INT:
                std::cout << value->integer;
                break;
            case JsonValue::TYPE_DEC:
                std::cout << value->decimal;
                break;
        }
    }

    /* EBNF of JSON:
     *
     * json            = value
     *
     * value           = string | number | object | array | "true" | "false" | "null"
     *
     * object          = "{" , object-elements, "}"
     * object-elements = string, ":", value, [ ",", object-elements ]
     *
     * array           = "[", array-elements, "]"
     * array-elements  = value, [ ",", array-elements ]
     *
     * string          = '"', { character }, '"'
     *
     * character       = "\", ( '"' | "\" | "/" | "b" | "f" | "n" | "r" | "t" | "u", digit, digit, digit, digit ) | unicode
     *
     * digit           = "0" | digit-1-9
     * digit-1-9       = "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
     * number          = [ "-" ], ( "0" | digit-1-9, { digit } ), [ ".", { digit } ], [ ( "e" | "E" ), ( "+" | "-" ), { digit } ]
     */

    inline void __error_maybe_escape(char *context, size_t *length, char ch) {
        if (ch == '\t' || ch == '\n' || ch == '\b' || ch == '\0') {
            context[(*length)++] = '\\';
//            context[(*length)++] = '[';
            switch (ch) {
                case '\t':
                    context[(*length)++] = 't';
                    break;
                case '\n':
                    context[(*length)++] = 'n';
                    break;
                case '\b':
                    context[(*length)++] = 'b';
                    break;
                case '\0':
                    context[(*length)++] = '0';
                    break;
                default:
                    break;
            }
//            context[(*length)++] = ']';
        } else {
            context[(*length)++] = ch;
        }
    }

    [[noreturn]] void __error(const std::string &message, const char *input, size_t offset) {
        static const size_t context_len = 20;
        char *context = new char[(2 * context_len + 1) * 4];  // add space for escaped chars
        size_t length = 0;
        if (offset > context_len) {
            context[0] = context[1] = context[2] = '.';
            length = 3;
        }
        for (size_t i = offset > context_len ? offset - context_len : 0U; i < offset; ++i)
            __error_maybe_escape(context, &length, input[i]);
        size_t left = length;
        bool end = false;
        for (size_t i = offset; i < offset + context_len; ++i) {
            if (input[i] == '\0') {
                end = true;
                break;
            }
            __error_maybe_escape(context, &length, input[i]);
        }
        if (!end) {
            context[length] = context[length + 1] = context[length + 2] = '.';
            length += 3;
        }
        context[length] = 0;
        std::stringstream stream;
        stream << message << std::endl;
        stream << "context: " << context << std::endl;
        delete[] context;
        stream << "         " << std::string(left, ' ') << "^";
        throw std::runtime_error(stream.str());
    }

    inline bool _all_digits(const char *s) {
        uint64_t val = *reinterpret_cast<const uint64_t *>(s);
        return (((val & 0xf0f0f0f0f0f0f0f0)
                 | (((val + 0x0606060606060606) & 0xf0f0f0f0f0f0f0f0) >> 4U)) == 0x3333333333333333);
    }

    inline uint32_t _parse_eight_digits(const char *s) {
        const __m128i ascii0 = _mm_set1_epi8('0');
        const __m128i mul_1_10 = _mm_setr_epi8(10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1);
        const __m128i mul_1_100 = _mm_setr_epi16(100, 1, 100, 1, 100, 1, 100, 1);
        const __m128i mul_1_10000 = _mm_setr_epi16(10000, 1, 10000, 1, 10000, 1, 10000, 1);
        __m128i in = _mm_sub_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(s)), ascii0);
        __m128i t1 = _mm_maddubs_epi16(in, mul_1_10);
        __m128i t2 = _mm_madd_epi16(t1, mul_1_100);
        __m128i t3 = _mm_packus_epi32(t2, t2);
        __m128i t4 = _mm_madd_epi16(t3, mul_1_10000);
        return _mm_cvtsi128_si32(t4);
    }

    std::variant<double, long long int> parse_number(const char *input, bool *is_decimal, size_t offset) {
        const char *s = input + offset;
        long long int integer = 0LL;
        double decimal = 0.0;
        bool negative = false, _is_decimal = false;
        if (*s == '-') {
            ++s;
            negative = true;
        }
        if (*s == '0') {
            ++s;
            if (*s >= '0' && *s <= '9')
                __error("numbers cannot have leading zeros", input, offset);
        } else {
#if PARSE_NUMBER_AVX
            while (_all_digits(s)) {
                integer = integer * 100000000 + _parse_eight_digits(s);
                s += 8;
            }
#endif
            while (*s >= '0' && *s <= '9')
                integer = integer * 10 + (*s++ - '0');
        }
        if (*s == '.') {
            _is_decimal = true;
            decimal = integer;
            double multiplier = 0.1;
            ++s;
#if PARSE_NUMBER_AVX
            while (_all_digits(s)) {
                decimal += _parse_eight_digits(s) * multiplier * 0.0000001;  // 7 digits
                multiplier *= 0.00000001;  // 8 digits
                s += 8;
            }
#endif
            while (*s >= '0' && *s <= '9') {
                decimal += (*s++ - '0') * multiplier;
                multiplier *= 0.1;
            }
        }
        if (*s == 'e' || *s == 'E') {
            if (!_is_decimal) {
                _is_decimal = true;
                decimal = integer;
            }
            ++s;
            bool negative_exp = false;
            if (*s == '-') {
                negative_exp = true;
                ++s;
            } else if (*s == '+') ++s;
            double exponent = 0.0;
            while (*s >= '0' && *s <= '9')
                exponent = exponent * 10.0 + (*s++ - '0');
            if (negative_exp) exponent = -exponent;
            decimal *= pow(10.0, exponent);
        }
        *is_decimal = _is_decimal;
        if (negative) {
            if (_is_decimal) return -decimal;
            else return -integer;
        } else {
            if (_is_decimal) return decimal;
            else return integer;
        }
    }

    void parse_str_naive(const char *src, char *dest, size_t *len, size_t offset) {
        bool escape = false;
        char *ptr = dest == nullptr ? const_cast<char *>(src) : dest, *base = ptr;
        for (const char *end = src + offset; escape || *end != '"'; ++end) {
            if (escape) {
                switch (*end) {
                    case '"':
                        *ptr++ = '"';
                        break;
                    case '\\':
                        *ptr++ = '\\';
                        break;
                    case '/':
                        *ptr++ = '/';
                        break;
                    case 'b':
                        *ptr++ = '\b';
                        break;
                    case 'f':
                        *ptr++ = '\f';
                        break;
                    case 'n':
                        *ptr++ = '\n';
                        break;
                    case 'r':
                        *ptr++ = '\r';
                        break;
                    case 't':
                        *ptr++ = '\t';
                        break;
                    case 'u':
                        // TODO: Should deal with unicode encoding
                        *ptr++ = '\\';
                        *ptr++ = 'u';
                        break;
                    default:
                        __error("invalid escape sequence", src, end - src);
                }
                escape = false;
            } else {
                if (*end == '\\') escape = true;
                else *ptr++ = *end;
            }
        }
        *ptr++ = 0;
        if (len != nullptr) *len = ptr - base;
    }

    bool parse_true(const char *s, size_t offset) {
        const auto *literal = reinterpret_cast<const uint32_t *>(s + offset);
        uint32_t target = 0x65757274;
        if (target != *literal || !kStructuralOrWhitespace[s[offset + 4]])
            __error("invalid true value", s, offset);
        return true;
    }

    bool parse_false(const char *s, size_t offset) {
        const auto *literal = reinterpret_cast<const uint64_t *>(s + offset);
        uint64_t target = 0x00000065736c6166;
        uint64_t mask = 0x000000ffffffffff;
        if (target != (*literal & mask) || !kStructuralOrWhitespace[s[offset + 5]])
            __error("invalid false value", s, offset);
        return false;
    }

    void parse_null(const char *s, size_t offset) {
        const auto *literal = reinterpret_cast<const uint32_t *>(s + offset);
        uint32_t target = 0x6c6c756e;
        if (target != *literal || !kStructuralOrWhitespace[s[offset + 4]])
            __error("invalid null value", s, offset);
    }

    [[noreturn]] void JSON::_error(const char *expected, char encountered, size_t index) {
        std::stringstream stream;
        char _encounter[3];
        size_t len = 0;
        __error_maybe_escape(_encounter, &len, encountered);
        _encounter[len] = 0;
        stream << "expected " << expected << " at index " << index << ", but encountered '" << _encounter << "'";
        MercuryJson::__error(stream.str(), input, index);
    }

#define next_char() ({ \
        idx = *idx_ptr++; \
        if (idx >= input_len) throw std::runtime_error("text ended prematurely"); \
        ch = input[idx]; \
    })

#define peek_char() ({ \
        idx = *idx_ptr; \
        ch = input[idx]; \
    })

#define expect(__char) ({ \
        if (ch != (__char)) _error(#__char, ch, idx); \
    })

#define error(__expected) ({ \
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

    inline char *JSON::_parse_str(size_t idx) {
#if ALLOC_PARSED_STR
        char *dest = literals + idx + 1;
#else
        char *dest = input + idx + 1;
#endif

        //@formatter:off
#if !PARSE_STR_NUM_THREADS
# if PARSE_STR_MODE == 2
        parse_str_per_bit
# elif PARSE_STR_MODE == 1
        parse_str_avx
# elif PARSE_STR_MODE == 0
        parse_str_naive
# endif
        (input, dest, nullptr, idx + 1);
#endif
        //@formatter:on
        return dest;
    }

#if PARSE_STR_NUM_THREADS

    void JSON::_thread_parse_str(size_t pid) {
//        auto start_time = std::chrono::steady_clock::now();
        size_t idx;
        const size_t *idx_ptr = indices + pid * num_indices / PARSE_STR_NUM_THREADS;  // deliberate shadowing
        const size_t *end_ptr = indices + (pid + 1) * num_indices / PARSE_STR_NUM_THREADS;
        char ch;
        do {
            peek_char();
            if (ch == '"') {
# if ALLOC_PARSED_STR
                char *dest = literals + idx + 1;
# else
                char *dest = input + idx + 1;
# endif

                //@formatter:off
# if PARSE_STR_MODE == 2
                parse_str_per_bit
# elif PARSE_STR_MODE == 1
                parse_str_avx
# elif PARSE_STR_MODE == 0
                parse_str_naive
# endif
                (input, dest, nullptr, idx + 1);
                //@formatter:on
            }
            ++idx_ptr;
        } while (idx_ptr != end_ptr);
//        std::chrono::duration<double> runtime = std::chrono::steady_clock::now() - start_time;
//        printf("parse str thread: %.6lf\n", runtime.count());
    }

#endif

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

    namespace shift_reduce_impl {
        struct JsonPartialValue;

        struct JsonPartialObject {
            const char *key;
            JsonPartialValue *value;
            JsonPartialObject *next;
            JsonPartialObject *final;

            explicit JsonPartialObject(const char *key, JsonPartialValue *value, JsonPartialObject *next = nullptr)
                    : key(key), value(value), next(next), final(this) {}
        };

        struct JsonPartialArray {
            JsonPartialValue *value;
            JsonPartialArray *next;
            JsonPartialArray *final;

            explicit JsonPartialArray(JsonPartialValue *value, JsonPartialArray *next = nullptr)
                    : value(value), next(next), final(this) {}
        };

        struct JsonPartialValue {
            enum ValueType : int {
                TYPE_NULL, TYPE_BOOL, TYPE_STR, TYPE_OBJ, TYPE_ARR, TYPE_INT, TYPE_DEC,
                TYPE_PARTIAL_OBJ, TYPE_PARTIAL_ARR, TYPE_CHAR
            } type;

            union {
                bool boolean;
                const char *str;
                JsonObject *object;
                JsonArray *array;
                JsonPartialObject *partial_object;
                JsonPartialArray *partial_array;
                long long int integer;
                double decimal;
                char structural;
            };

            //@formatter:off
            explicit JsonPartialValue() : type(TYPE_NULL) {}
            explicit JsonPartialValue(bool value) : type(TYPE_BOOL), boolean(value) {}
            explicit JsonPartialValue(const char *value) : type(TYPE_STR), str(value) {}
            explicit JsonPartialValue(JsonObject *value) : type(TYPE_OBJ), object(value) {}
            explicit JsonPartialValue(JsonArray *value) : type(TYPE_ARR), array(value) {}
            explicit JsonPartialValue(long long int value) : type(TYPE_INT), integer(value) {}
            explicit JsonPartialValue(double value) : type(TYPE_DEC), decimal(value) {}

            explicit JsonPartialValue(JsonPartialObject *value) : type(TYPE_PARTIAL_OBJ), partial_object(value) {}
            explicit JsonPartialValue(JsonPartialArray *value) : type(TYPE_PARTIAL_ARR), partial_array(value) {}
            explicit JsonPartialValue(char value) : type(TYPE_CHAR), structural(value) {}
            //@formatter:on
        };

        void print_json(JsonPartialValue *value, size_t indent = 0) {
            switch (value->type) {
                case JsonPartialValue::TYPE_PARTIAL_OBJ:
                case JsonPartialValue::TYPE_PARTIAL_ARR: {
                    print_indent(indent);
                    std::cout << "(partial) ";
                    JsonValue new_value = *reinterpret_cast<JsonValue *>(value);
                    new_value.type = static_cast<JsonValue::ValueType>(static_cast<int>(value->type) - 4);
                    print_json(&new_value, indent);
                }
                case JsonPartialValue::TYPE_NULL:
                case JsonPartialValue::TYPE_BOOL:
                case JsonPartialValue::TYPE_STR:
                case JsonPartialValue::TYPE_OBJ:
                case JsonPartialValue::TYPE_ARR:
                case JsonPartialValue::TYPE_INT:
                case JsonPartialValue::TYPE_DEC:
                    print_json(reinterpret_cast<JsonValue *>(value), indent);
                    break;
                case JsonPartialValue::TYPE_CHAR:
                    print_indent(indent);
                    std::cout << "'" << value->structural << "'";
                    break;
            }
            std::cout << std::endl;
        }

        static const size_t kDefaultStackSize = 4096;

        class ParseStack {
            JsonPartialValue **stack;
            size_t stack_top;
            BlockAllocator<JsonValue> &allocator;

        public:
            ParseStack(BlockAllocator<JsonValue> &allocator, size_t max_size = kDefaultStackSize)
                    : allocator(allocator) {
                stack = static_cast<JsonPartialValue **>(aligned_malloc(max_size * sizeof(JsonPartialValue *)));
                stack_top = 0;
            }

            ParseStack(ParseStack &&other) noexcept : allocator(other.allocator) {
                stack = other.stack;
                stack_top = other.stack_top;
                other.stack = nullptr;
            }

            ~ParseStack() {
                aligned_free(stack);
            }

            inline size_t size() const { return stack_top; }

            inline JsonPartialValue *operator [](size_t idx) const { return stack[idx]; }

            template <typename ...Args>
            inline bool _check_stack_top(Args ...) const;

            template <typename ...Args>
            inline bool _check_stack_top(bool, Args ...) const;

            template <typename ...Args>
            inline bool _check_stack_top(char, Args ...) const;

            template <typename ...Args>
            inline bool _check_stack_top(JsonPartialValue::ValueType, Args ...) const;

            template <typename ...Args>
            inline bool check(Args ...args) const {
                if (stack_top < sizeof...(args)) return false;
                return _check_stack_top(args...);
            }

            inline bool check_pos(size_t pos, char ch) const {
                // pos starts from 1, counts from stack top.
                if (stack_top < pos) return false;
                auto *top = stack[stack_top - pos];
                return top->type == JsonPartialValue::TYPE_CHAR && top->structural == ch;
            }

            inline bool check_pos(size_t pos, JsonPartialValue::ValueType type) const {
                // pos starts from 1, counts from stack top.
                if (stack_top < pos) return false;
                auto *top = stack[stack_top - pos];
                return top->type == type;
            }

            inline JsonPartialValue *get(size_t pos) const {
                return stack[stack_top - pos];
            }

            template <typename ...Args>
            inline void push(Args ...args) {
                stack[stack_top++] = allocator.construct<JsonPartialValue>(std::forward<Args>(args)...);
            }

            inline void push(JsonPartialValue *value) {
                stack[stack_top++] = value;
            }

            inline void pop(size_t n) {
                stack_top -= n;
            }

            void print() {
                std::cout << "Stack size: " << stack_top << std::endl;
                for (size_t i = 0; i < stack_top; ++i) {
                    std::cout << "Element #" << i << ": ";
                    print_json(stack[i]);
                }
            }

            // "{", partial-object, "}"  =>  object
            inline void reduce_object() {
                if (check('{')) {
                    // Emtpy object.
                    pop(1);
                    push(static_cast<JsonObject *>(nullptr));
                } else if (check('{', JsonPartialValue::TYPE_PARTIAL_OBJ)) {
                    // Non-empty object.
                    auto *obj = reinterpret_cast<JsonObject *>(get(1)->partial_object);
                    pop(2);
                    push(obj);
                } else {
                    // This should not happen when the input is well-formed.
                    push('}');
                }
            }

            // "[", partial-array, "]"  =>  array
            inline void reduce_array() {
                if (check('[')) {
                    // Emtpy array.
                    pop(1);
                    push(static_cast<JsonArray *>(nullptr));
                } else if (check('[', true)) {
                    // Construct singleton array.
                    auto *arr = allocator.construct<JsonArray>(reinterpret_cast<JsonValue *>(get(1)), nullptr);
                    pop(2);
                    push(arr);
                } else if (check('[', JsonPartialValue::TYPE_PARTIAL_ARR, ',', true)) {
                    // We have to manually match the final element in the array, because we only reduce to partial
                    // array on commas (,).
                    auto *partial_arr = get(3)->partial_array;
                    partial_arr->final->next = reinterpret_cast<JsonPartialArray *>(
                            allocator.construct<JsonArray>(reinterpret_cast<JsonValue *>(get(1)), nullptr));
                    auto *arr = reinterpret_cast<JsonArray *>(partial_arr);
                    pop(4);
                    push(arr);
                } else {
                    // This should not happen when the input is well-formed.
                    push(']');
                }
            }

            // ( [ partial-array ], "," | "[" ), value, ","  =>  partial-array, ","
            inline void reduce_partial_array() {
                if (check('[', true)) {
                    // Construct a singleton partial array. Note that previous value must be of complete type,
                    // otherwise we might aggressively match partial objects.
                    auto *elem = get(1);
                    auto *partial_arr = allocator.construct<JsonPartialArray>(elem, nullptr);
                    pop(1);
                    push(partial_arr);
                    push(',');
                } else if (check(',', true)) {
                    // Merge with previous partial array.
                    auto *elem = get(1);
                    auto *partial_arr = allocator.construct<JsonPartialArray>(elem, nullptr);
                    if (check_pos(3, JsonPartialValue::TYPE_PARTIAL_ARR)) {
                        auto *prev_arr = get(3)->partial_array;
                        prev_arr->final = prev_arr->final->next = partial_arr;
                        pop(1);  // No need to push ',' --- just re-use the previous one.
                    } else {
                        pop(1);
                        push(partial_arr);
                        push(',');
                    }
                } else {
                    push(',');
                }
            }

            // [ partial-object, "," ], string, ":", value  =>  partial-object
            inline bool reduce_partial_object() {
                if (check(JsonPartialValue::TYPE_STR, ':', true)) {
                    // Construct singleton partial object.
                    auto *key = get(3)->str;
                    auto *value = get(1);
                    auto *partial_obj = allocator.construct<JsonPartialObject>(key, value, nullptr);
                    pop(3);
                    if (check(JsonPartialValue::TYPE_PARTIAL_OBJ, ',')) {
                        // Merge with previous partial object.
                        auto *prev_obj = get(2)->partial_object;
                        prev_obj->final = prev_obj->final->next = partial_obj;
                        partial_obj = prev_obj;
                        pop(2);
                    }
                    push(partial_obj);
                    return true;
                }
                return false;
            }
        };

        template <>
        inline bool ParseStack::_check_stack_top<>() const { return true; }

        // true for any fully-parsed JSON value
        template <typename ...Args>
        inline bool ParseStack::_check_stack_top(bool first, Args ...args) const {
//            assert(first);
            auto *top = stack[stack_top - sizeof...(args) - 1];
            switch (top->type) {
                case JsonPartialValue::TYPE_PARTIAL_OBJ:
                case JsonPartialValue::TYPE_PARTIAL_ARR:
                case JsonPartialValue::TYPE_CHAR:
                    return false;
                default:
                    break;
            }
            return _check_stack_top(args...);
        }

        template <typename ...Args>
        inline bool ParseStack::_check_stack_top(char first, Args ...args) const {
            auto *top = stack[stack_top - sizeof...(args) - 1];
            if (top->type != JsonPartialValue::TYPE_CHAR || top->structural != first) return false;
            return _check_stack_top(args...);
        }

        template <typename ...Args>
        inline bool ParseStack::_check_stack_top(JsonPartialValue::ValueType first, Args ...args) const {
            auto *top = stack[stack_top - sizeof...(args) - 1];
            if (top->type != first) return false;
            return _check_stack_top(args...);
        }

    }

    void JSON::_thread_shift_reduce_parsing(const size_t *idx_begin, const size_t *idx_end,
                                            shift_reduce_impl::ParseStack *stack) {
        using shift_reduce_impl::JsonPartialValue;
        using shift_reduce_impl::JsonPartialObject;
        using shift_reduce_impl::JsonPartialArray;
        using shift_reduce_impl::ParseStack;

        while (idx_begin != idx_end) {
            size_t idx = *idx_begin;
            char ch = input[idx];

            // Shift current value onto stack.
            switch (ch) {
                case '"':
                    stack->push(_parse_str(idx));
                    break;
                case 't':
                    stack->push(parse_true(input, idx));
                    break;
                case 'f':
                    stack->push(parse_false(input, idx));
                    break;
                case 'n':
                    parse_null(input, idx);
                    stack->push();
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
                    if (is_decimal) {
                        stack->push(std::get<double>(ret));
                    } else {
                        stack->push(std::get<long long int>(ret));
                    }
                    break;
                }

                case '{':
                case '[':
                case ':':
                    stack->push(ch);
                    break;

                    // Perform reduce on the stack.
                    // There will be at most two consecutive reduce ops:
                    //   1. From a partial array to array, or from a partial object to object.
                    //   2. Merge the previously constructed value with a partial array or partial object.
                case '}':
                    // "{", partial-object, "}"  =>  object
                    stack->reduce_object();
                    break;
                case ']':
                    // "[", partial-array, "]"  =>  array
                    stack->reduce_array();
                    break;
                case ',':
                    // ( [ partial-array ], "," | "[" ), value, ","  =>  partial-array, ","
                    stack->reduce_partial_array();
                    break;
                default:
                    error("JSON value");
            }

            // Possible second step of reduce.
            stack->reduce_partial_object();

            ++idx_begin;
        }
    }

    JsonValue *JSON::_shift_reduce_parsing() {
        using shift_reduce_impl::JsonPartialValue;
        using shift_reduce_impl::ParseStack;
#if SHIFT_REDUCE_NUM_THREADS > 1
        std::thread shift_reduce_threads[SHIFT_REDUCE_NUM_THREADS - 1];
        std::vector<BlockAllocator<JsonValue>> allocators;
        std::vector<ParseStack> stacks;
        stacks.emplace_back(allocator);
        size_t num_indices_per_thread = (num_indices - 1 + SHIFT_REDUCE_NUM_THREADS) / SHIFT_REDUCE_NUM_THREADS;
        for (int i = 0; i < SHIFT_REDUCE_NUM_THREADS - 1; ++i)
            allocators.push_back(allocator.fork(2 * num_indices_per_thread * sizeof(JsonPartialValue)));
        for (int i = 0; i < SHIFT_REDUCE_NUM_THREADS - 1; ++i)
            stacks.emplace_back(allocators[i]);
        for (int i = 0; i < SHIFT_REDUCE_NUM_THREADS - 1; ++i) {
            size_t idx_begin = (num_indices - 1) * (i + 1) / SHIFT_REDUCE_NUM_THREADS;
            size_t idx_end = (num_indices - 1) * (i + 2) / SHIFT_REDUCE_NUM_THREADS;
            shift_reduce_threads[i] = std::thread(&JSON::_thread_shift_reduce_parsing, this,
                                                  indices + idx_begin, indices + idx_end, &stacks[i + 1]);
        }
        size_t idx_end = (num_indices - 1) / SHIFT_REDUCE_NUM_THREADS;
        _thread_shift_reduce_parsing(indices, indices + idx_end, &stacks[0]);
        // Join threads and merge.
        ParseStack &main_stack = stacks[0];
//        main_stack.print();
        for (int i = 0; i < SHIFT_REDUCE_NUM_THREADS - 1; ++i) {
            shift_reduce_threads[i].join();
            ParseStack &merge_stack = stacks[i + 1];
//            merge_stack.print();
            for (size_t idx = 0; idx < merge_stack.size(); ++idx) {
                auto *value = merge_stack[idx];
                switch (value->type) {
                    case JsonPartialValue::TYPE_NULL:
                    case JsonPartialValue::TYPE_BOOL:
                    case JsonPartialValue::TYPE_STR:
                    case JsonPartialValue::TYPE_OBJ:
                    case JsonPartialValue::TYPE_ARR:
                    case JsonPartialValue::TYPE_INT:
                    case JsonPartialValue::TYPE_DEC:
                        main_stack.push(value);
                        break;
                    case JsonPartialValue::TYPE_PARTIAL_OBJ:
                        if (main_stack.check(JsonPartialValue::TYPE_PARTIAL_OBJ, ',')) {
                            // Merge with partial object from previous stack.
                            auto *prev_obj = main_stack.get(2)->partial_object;
                            prev_obj->final->next = value->partial_object;
                            prev_obj->final = value->partial_object->final;
                            main_stack.pop(1);
                        } else {
                            main_stack.push(value);
                        }
                        break;
                    case JsonPartialValue::TYPE_PARTIAL_ARR:
                        if (main_stack.check(JsonPartialValue::TYPE_PARTIAL_ARR, ',')) {
                            // Merge with partial array from previous stack.
                            auto *prev_arr = main_stack.get(2)->partial_array;
                            prev_arr->final->next = value->partial_array;
                            prev_arr->final = value->partial_array->final;
                            main_stack.pop(1);
                        } else {
                            main_stack.push(value);
                        }
                        break;
                    case JsonPartialValue::TYPE_CHAR:
                        switch (char ch = value->structural) {
                            case '{':
                            case '[':
                            case ':':
                                main_stack.push(ch);
                                break;

                                // Perform reduce on the stack.
                                // There will be at most two consecutive reduce ops:
                                //   1. From a partial array to array, or from a partial object to object.
                                //   2. Merge the previously constructed value with a partial array or partial object.
                            case '}':
                                // "{", partial-object, "}"  =>  object
                                main_stack.reduce_object();
                                break;
                            case ']':
                                // "[", partial-array, "]"  =>  array
                                main_stack.reduce_array();
                                break;
                            case ',':
                                // value, ","  =>  partial-array
                                main_stack.reduce_partial_array();
                                break;
                            default:
                                error("JSON value");
                        }
                        break;
                }
                while (main_stack.reduce_partial_object()) {}
            }
        }
#else
        ParseStack main_stack(allocator);
        _thread_shift_reduce_parsing(indices, indices + num_indices - 1, &main_stack);
#endif
//        main_stack.print();
        assert(main_stack.size() == 1);
        auto *ret = reinterpret_cast<JsonValue *>(main_stack[0]);
        idx_ptr += num_indices - 1;  // Consume the indices to satisfy null ending check.
        return ret;
    }

    JSON::JSON(char *document, size_t size, bool manual_construct) : allocator(size) {
        input = document;
        input_len = size;
        this->document = nullptr;

        idx_ptr = indices = static_cast<size_t *>(aligned_malloc(size * sizeof(size_t)));
        num_indices = 0;
#if ALLOC_PARSED_STR
        literals = static_cast<char *>(aligned_malloc(size));
#endif

        if (!manual_construct) {
            exec_stage1();
            exec_stage2();
        }
    }

    void JSON::exec_stage1() {
        uint64_t prev_escape_mask = 0;
        uint64_t prev_quote_mask = 0;
        uint64_t prev_pseudo_mask = 0;
        for (size_t offset = 0; offset < input_len; offset += 64) {
            __m256i _input1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + offset));
            __m256i _input2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + offset + 32));
            Warp warp(_input2, _input1);
            uint64_t escape_mask = extract_escape_mask(warp, &prev_escape_mask);
            uint64_t quote_mask = 0;
            uint64_t literal_mask = extract_literal_mask(warp, escape_mask, &prev_quote_mask, &quote_mask);
            uint64_t structural_mask = 0, whitespace_mask = 0;
            extract_structural_whitespace_characters(warp, literal_mask, &structural_mask, &whitespace_mask);
            uint64_t pseudo_mask = extract_pseudo_structural_mask(
                    structural_mask, whitespace_mask, quote_mask, literal_mask, &prev_pseudo_mask);
            construct_structural_character_pointers(pseudo_mask, offset, indices, &num_indices);
        }
    }

    void JSON::exec_stage2() {
#if PARSE_STR_NUM_THREADS
        std::thread parse_str_threads[PARSE_STR_NUM_THREADS];
        for (size_t i = 0; i < PARSE_STR_NUM_THREADS; ++i)
            parse_str_threads[i] = std::thread(&JSON::_thread_parse_str, this, i);
#endif

#if SHIFT_REDUCE_PARSER
        document = _shift_reduce_parsing();  // final index is '\0'
#else
        document = _parse_value();
#endif
        size_t idx;
        char ch;
        peek_char();
        if (ch != 0) error("file end");
#if PARSE_STR_NUM_THREADS
        for (std::thread &thread : parse_str_threads)
            thread.join();
#endif
        aligned_free(indices);
        indices = nullptr;
    }

#undef next_char
#undef peek_char
#undef expect
#undef error

    JSON::~JSON() {
        if (indices != nullptr) aligned_free(indices);
#if ALLOC_PARSED_STR
        aligned_free(literals);
#endif
    }

    void parse_str_per_bit(const char *src, char *dest, size_t *len, size_t offset) {
        const char *_src = src;
        src += offset;
        char *base = dest;
        while (true) {
            __m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest), input);
            __mmask32 backslash_mask = __cmpeq_mask<'\\'>(input);
            __mmask32 quote_mask = __cmpeq_mask<'"'>(input);

            if (((backslash_mask - 1) & quote_mask) != 0) {
                // quotes first
                size_t quote_offset = _tzcnt_u32(quote_mask);
                dest[quote_offset] = 0;
                if (len != nullptr) *len = (dest - base) + quote_offset;
                break;
            } else if (((quote_mask - 1) & backslash_mask) != 0) {
                // backslash first
                size_t backslash_offset = _tzcnt_u32(backslash_mask);
                uint8_t escape_char = src[backslash_offset + 1];
                if (escape_char == 'u') {
                    // TODO: deal with unicode characters
                    memcpy(dest + backslash_offset, src + backslash_offset, 6);
                    src += backslash_offset + 6;
                    dest += backslash_offset + 6;
                } else {
                    uint8_t escaped = kEscapeMap[escape_char];
                    if (escaped == 0U)
                        __error("invalid escape character '" + std::string(1, escape_char) + "'", _src, offset);
                    dest[backslash_offset] = kEscapeMap[escape_char];
                    src += backslash_offset + 2;
                    dest += backslash_offset + 1;
                }
            } else {
                // nothing here
                src += sizeof(__mmask32);
                dest += sizeof(__mmask32);
            }
        }
    }

    void parse_str_avx(const char *src, char *dest, size_t *len, size_t offset) {
#if PARSE_STR_32BIT
        typedef __mmask32 mask_t;
# define tzcnt _tzcnt_u32
# define blsr _blsr_u32
#else
        typedef uint64_t mask_t;
# define tzcnt _tzcnt_u64
# define blsr _blsr_u64
#endif

        const char *_src = src;
        src += offset;
        if (dest == nullptr) dest = const_cast<char *>(src);
        char *base = dest;
        mask_t prev_odd_backslash_ending_mask = 0ULL;
        while (true) {
#if PARSE_STR_32BIT
            __m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
#else
            Warp input(src);
#endif
            mask_t escape_mask = extract_escape_mask(input, &prev_odd_backslash_ending_mask);
            mask_t quote_mask = __cmpeq_mask<'"'>(input) & (~escape_mask);

            size_t ending_offset = tzcnt(quote_mask);

            if (ending_offset < sizeof(mask_t) * 8) {
                size_t last_offset = 0, length;
                while (true) {
                    size_t this_offset = tzcnt(escape_mask);
                    if (this_offset >= ending_offset) {
                        memmove(dest, src + last_offset, ending_offset - last_offset);
                        dest[ending_offset - last_offset] = '\0';
                        if (len != nullptr) *len = (dest - base) + (ending_offset - last_offset);
                        break;
                    }
                    length = this_offset - last_offset;
                    char escaper = src[this_offset];
                    memmove(dest, src + last_offset, length);
                    dest += length;
                    *(dest - 1) = kEscapeMap[escaper];
                    last_offset = this_offset + 1;
                    escape_mask = blsr(escape_mask);
                }
                break;
            } else {
#if PARSE_STR_FULLY_AVX
                /* fully-AVX version */
                __m256i lo_mask = convert_to_mask(escape_mask);
                __m256i hi_mask = convert_to_mask(escape_mask >> 32U);
                // mask ? translated : original
                __m256i lo_trans = translate_escape_characters(input.lo);
                __m256i hi_trans = translate_escape_characters(input.hi);
                input.lo = _mm256_blendv_epi8(lo_trans, input.lo, lo_mask);
                input.hi = _mm256_blendv_epi8(hi_trans, input.hi, hi_mask);
                uint64_t escaper_mask = (escape_mask >> 1U) | (prev_odd_backslash_ending_mask << 63U);

                deescape(input, escaper_mask);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest), input.lo);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest + 32), input.hi);
                dest += 64 - _mm_popcnt_u64(escaper_mask);
                src += 64;
#else
                size_t last_offset = 0, length;
                while (true) {
                    size_t this_offset = tzcnt(escape_mask);
                    length = this_offset - last_offset;
                    char escaper = src[this_offset];
                    memmove(dest, src + last_offset, length);
                    dest += length;
                    if (this_offset >= ending_offset) break;
                    *(dest - 1) = kEscapeMap[escaper];
                    last_offset = this_offset + 1;
                    escape_mask = blsr(escape_mask);
                }
                src += sizeof(mask_t) * 8;
#endif
            }
        }
#undef tzcnt
#undef blsr
    }

    inline __m256i convert_to_mask(uint32_t input) {
        /* Create a mask based on each bit of `input`.
         *                    input : [0-31]
         * _mm256_set1_epi32(input) : [0-7] [8-15] [16-23] [24-31] * 8
         *      _mm256_shuffle_epi8 : [0-7] * 8 [8-15] * 8 ...
         *         _mm256_and_si256 : [0] [1] [2] [3] [4] ...
         *        _mm256_cmpeq_epi8 : 0xFF for 1-bits, 0x00 for 0-bits
         */
        const __m256i projector = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                                   1, 1, 1, 1, 1, 1, 1, 1,
                                                   2, 2, 2, 2, 2, 2, 2, 2,
                                                   3, 3, 3, 3, 3, 3, 3, 3);
        const __m256i masker = _mm256_setr_epi8(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
        const __m256i zeros = _mm256_set1_epi8(0);
        return _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(
                _mm256_set1_epi32(input), projector), masker), zeros);
    }

    inline uint64_t __extract_highestbit_pext(const Warp &input, int shift, uint64_t escaper_mask) {
        uint64_t lo = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(input.lo, shift)));
        uint64_t hi = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(input.hi, shift)));
        return _pext_u64(((hi << 32U) | lo), escaper_mask);
    }

    inline __m256i __expand(uint32_t input) {
        /* 0x00 for 1-bits, 0x01 for 0-bits */
        const __m256i ones = _mm256_set1_epi8(1);
        return _mm256_add_epi8(convert_to_mask(input), ones);
    }

    inline __m256i __reconstruct(uint32_t h0, uint32_t h1, uint32_t h2, uint32_t h3,
                                 uint32_t h4, uint32_t h5, uint32_t h6, uint32_t h7) {
        /* Reconstruct a __m256i with 8-bit numbers specified by bits from h0~h7 */
        __m256i result = __expand(h0);
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h1), 1));
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h2), 2));
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h3), 3));
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h4), 4));
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h5), 5));
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h6), 6));
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h7), 7));
        return result;
    }

    void deescape(Warp &input, uint64_t escaper_mask) {
        /* Remove 8-bit characters specified by `escaper_mask` */
        uint64_t nonescaper_mask = ~escaper_mask;
        // Obtain each bit from each 8-bit character, keeping only non-escapers
        uint64_t h7 = __extract_highestbit_pext(input, 0, nonescaper_mask);
        uint64_t h6 = __extract_highestbit_pext(input, 1, nonescaper_mask);
        uint64_t h5 = __extract_highestbit_pext(input, 2, nonescaper_mask);
        uint64_t h4 = __extract_highestbit_pext(input, 3, nonescaper_mask);
        uint64_t h3 = __extract_highestbit_pext(input, 4, nonescaper_mask);
        uint64_t h2 = __extract_highestbit_pext(input, 5, nonescaper_mask);
        uint64_t h1 = __extract_highestbit_pext(input, 6, nonescaper_mask);
        uint64_t h0 = __extract_highestbit_pext(input, 7, nonescaper_mask);
        // _mm256_packus_epi32
        input.lo = __reconstruct(h0, h1, h2, h3, h4, h5, h6, h7);
        input.hi = __reconstruct(
                h0 >> 32U, h1 >> 32U, h2 >> 32U, h3 >> 32U, h4 >> 32U, h5 >> 32U, h6 >> 32U, h7 >> 32U);
    }

    // 1. '/'   0x2f 0x2f
    // 2. '""'  0x22 0x22
    // 4. '\'   0x5c 0x5c
    // 8. 'b'   0x62 0x08
    // 16.'f'   0x66 0x0c
    // 32.'n'   0x6e 0x0a
    // 64.'r'   0x72 0x0d
    //128.'t'   0x74 0x09
    // TODO: Unicode literals and non-escapable character validation
    __m256i translate_escape_characters(__m256i input) {
        const __m256i hi_lookup = _mm256_setr_epi8(0, 0, 0x03, 0, 0, 0x04, 0x38, 0xc0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0x03, 0, 0, 0x04, 0x38, 0xc0, 0, 0, 0, 0, 0, 0, 0, 0);
        const __m256i lo_lookup = _mm256_setr_epi8(0, 0, 0x4a, 0, 0x80, 0, 0x10, 0, 0, 0, 0, 0, 0x04, 0, 0x20, 0x01,
                                                   0, 0, 0x4a, 0, 0x80, 0, 0x10, 0, 0, 0, 0, 0, 0x04, 0, 0x20, 0x01);
        __m256i hi_index = _mm256_shuffle_epi8(hi_lookup,
                                               _mm256_and_si256(_mm256_srli_epi16(input, 4), _mm256_set1_epi8(0x7F)));
        __m256i lo_index = _mm256_shuffle_epi8(lo_lookup, input);
        __m256i character_class = _mm256_and_si256(hi_index, lo_index);
        const __m256i trans_1 = _mm256_setr_epi8(0, 0x2f, 0x22, 0, 0x5c, 0, 0, 0, 0x08, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0x2f, 0x22, 0, 0x5c, 0, 0, 0, 0x08, 0, 0, 0, 0, 0, 0, 0);
        const __m256i trans_2 = _mm256_setr_epi8(0, 0x0c, 0x0a, 0, 0x0d, 0, 0, 0, 0x09, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0x0c, 0x0a, 0, 0x0d, 0, 0, 0, 0x09, 0, 0, 0, 0, 0, 0, 0);
        __m256i trans = _mm256_or_si256(
                _mm256_shuffle_epi8(trans_1, character_class),
                _mm256_shuffle_epi8(trans_2, _mm256_and_si256(
                        _mm256_srli_epi16(character_class, 4), _mm256_set1_epi8(0x7F))));
        return trans;
    }
}
