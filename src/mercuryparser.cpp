#include <cstdio>
#include <cstring>
#include <immintrin.h>

#include <deque>
#include <map>
#include <stack>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <bitset>
#include <algorithm>

#include "mercuryparser.h"

#define STATIC_CMPEQ_MASK 0
#define USE_PARSE_STR_AVX 1
#define PARSE_STR_FULL_AVX 1

namespace MercuryJson {

    template <char c>
    inline u_int64_t __cmpeq_mask(const Warp &raw) {
#if STATIC_CMPEQ_MASK
        static const __m256i vec_c = _mm256_set1_epi8(c);
#else
        const __m256i vec_c = _mm256_set1_epi8(c);
#endif
        u_int64_t hi = static_cast<u_int32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(raw.hi, vec_c)));
        u_int64_t lo = static_cast<u_int32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(raw.lo, vec_c)));
        return (hi << 32U) | lo;
    }

    void __print(Warp &raw) {
        auto *vals = reinterpret_cast<u_int8_t *>(&raw);
        for (size_t i = 0; i < 64; ++i) printf("%2x ", vals[i]);
        printf("\n");
    }

    void __printChar(Warp &raw) {
        auto *vals = reinterpret_cast<u_int8_t *>(&raw);
        for (size_t i = 0; i < 64; ++i) printf("%2x(%c) ", vals[i], vals[i]);
        printf("\n");
    }

    const u_int64_t __even_mask64 = 0x5555555555555555U;
    const u_int64_t __odd_mask64 = ~__even_mask64;

    // @formatter:off
    template <bool include_backslash>
    u_int64_t extract_escape_mask(const Warp &raw, u_int64_t *prev_odd_backslash_ending_mask) {
        u_int64_t backslash_mask = __cmpeq_mask<'\\'>(raw);
        u_int64_t start_backslash_mask = backslash_mask & ~(backslash_mask << 1U);
        u_int64_t even_start_backslash_mask = (start_backslash_mask & __even_mask64) ^ *prev_odd_backslash_ending_mask;
        u_int64_t even_carrier_backslash_mask = even_start_backslash_mask + backslash_mask;
        u_int64_t even_escape_mask;
        if (include_backslash) even_escape_mask = (even_carrier_backslash_mask ^ backslash_mask) & __odd_mask64;
        else even_escape_mask = even_carrier_backslash_mask & ~backslash_mask & __odd_mask64;

        u_int64_t odd_start_backslash_mask = (start_backslash_mask & __odd_mask64) ^ *prev_odd_backslash_ending_mask;
        u_int64_t odd_carrier_backslash_mask = odd_start_backslash_mask + backslash_mask;
        u_int64_t odd_backslash_ending_mask = odd_carrier_backslash_mask < odd_start_backslash_mask;
        *prev_odd_backslash_ending_mask = odd_backslash_ending_mask;
        u_int64_t odd_escape_mask;
        if (include_backslash) odd_escape_mask = (odd_carrier_backslash_mask ^ backslash_mask) & __even_mask64;
        else odd_escape_mask = odd_carrier_backslash_mask & ~backslash_mask & __even_mask64;
        return even_escape_mask | odd_escape_mask;
    }
    // @formatter:on

    // explicit instantiation
    template u_int64_t extract_escape_mask<true>(const Warp &raw, u_int64_t *prev_odd_backslash_ending_mask);
    template u_int64_t extract_escape_mask<false>(const Warp &raw, u_int64_t *prev_odd_backslash_ending_mask);

    u_int64_t extract_literal_mask(
            const Warp &raw, u_int64_t escape_mask, u_int64_t *prev_literal_ending, u_int64_t *quote_mask) {
        *quote_mask = __cmpeq_mask<'"'>(raw) & ~escape_mask;
        u_int64_t literal_mask = _mm_cvtsi128_si64(
                _mm_clmulepi64_si128(_mm_set_epi64x(0ULL, *quote_mask), _mm_set1_epi8(0xFF), 0));
        u_int64_t literal_reversor = *prev_literal_ending * ~0ULL;
        literal_mask ^= literal_reversor;
        *prev_literal_ending = literal_mask >> 63U;
        return literal_mask;
    }

    void extract_structural_whitespace_characters(
            const Warp &raw, u_int64_t literal_mask, u_int64_t *structural_mask, u_int64_t *whitespace_mask) {
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

    u_int64_t extract_pseudo_structural_mask(
            u_int64_t structural_mask, u_int64_t whitespace_mask, u_int64_t quote_mask, u_int64_t literal_mask,
            u_int64_t *prev_pseudo_structural_end_mask) {
        u_int64_t st_ws = structural_mask | whitespace_mask;
        structural_mask |= quote_mask;
        u_int64_t pseudo_structural_mask = (st_ws << 1U) | *prev_pseudo_structural_end_mask;
        *prev_pseudo_structural_end_mask = (st_ws >> 63U) & 1ULL;
        pseudo_structural_mask &= (~whitespace_mask) & (~literal_mask);
        structural_mask |= pseudo_structural_mask;
        structural_mask &= ~(quote_mask & ~literal_mask);
        return structural_mask;
    }

    void construct_structural_character_pointers(
            u_int64_t pseudo_structural_mask, size_t offset, size_t *indices, size_t *base) {
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

    const __mmask32 __even_mask = 0x55555555U;
    const __mmask32 __odd_mask = ~__even_mask;

    template <char c>
    inline __mmask32 __cmpeq_mask(__m256i raw) {
#if STATIC_CMPEQ_MASK
        static __m256i vec_c = _mm256_set1_epi8(c);
#else
        __m256i vec_c = _mm256_set1_epi8(c);
#endif
        return static_cast<__mmask32>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(raw, vec_c)));
    }

    void __print_m256i(__m256i raw) {
        u_int8_t *vals = reinterpret_cast<u_int8_t *>(&raw);
        for (size_t i = 0; i < 32; ++i) printf("%2x ", vals[i]);
        printf("\n");
    }

    void __printChar_m256i(__m256i raw) {
        u_int8_t *vals = reinterpret_cast<u_int8_t *>(&raw);
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

    const u_int64_t structural_or_whitespace[256] = {
            1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    const u_int8_t escape_map[256] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0x22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x2f,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x5c, 0, 0, 0,
            0, 0, 0x08, 0, 0, 0, 0x0c, 0, 0, 0, 0, 0, 0, 0, 0x0a, 0,
            0, 0, 0x0d, 0, 0x09, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

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

    inline JsonValue parseNumber(char *s) {
        long long int integer = 0LL;
        double decimal = 0.0;
        bool negative = false, is_decimal = false;
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
        if (negative) integer = -integer;
        if (*s == '.') {
            is_decimal = true;
            decimal = integer;
            double multiplier = 0.1;
            while (*s >= '0' && *s <= '9') {
                decimal += (*s++ - '0') * multiplier;
                multiplier /= 10.0;
            }
        }
        if (*s == 'e' || *s == 'E') {
            if (!is_decimal) {
                is_decimal = true;
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
        if (is_decimal) return JsonValue(decimal);
        else return JsonValue(integer);
    }

    char *parseStr(char *s) {
        bool escape = false;
        char *ptr = s + 1;
        for (char *end = s + 1; escape || *end != '"'; ++end) {
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
                        throw std::runtime_error("invalid escape sequence");
                }
                escape = false;
            } else {
                if (*end == '\\') escape = true;
                else *ptr++ = *end;
            }
        }
        *ptr++ = 0;
        return s + 1;
    }

    bool parseTrue(const char *s) {
        u_int64_t local = 0, mask = 0x0000000065757274;
        std::memcpy(&local, s, 4);
        if (mask != local || !structural_or_whitespace[s[4]])
            throw std::runtime_error("invalid true value");
        return true;
    }

    bool parseFalse(const char *s) {
        u_int64_t local = 0, mask = 0x00000065736c6166;
        std::memcpy(&local, s, 5);
        if (mask != local || !structural_or_whitespace[s[5]])
            throw std::runtime_error("invalid false value");
        return false;
    }

    void parseNull(const char *s) {
        u_int64_t local = 0, mask = 0x000000006c6c756e;
        std::memcpy(&local, s, 4);
        if (mask != local || !structural_or_whitespace[s[4]])
            throw std::runtime_error("invalid null value");
        return;
    }

    JsonValue *get_last(std::deque<JsonValue> &values) {
        return values.empty() ? nullptr : &values.back();
    }

    const JsonValue *get_last(const std::deque<JsonValue> &values) {
        return values.empty() ? nullptr : &values.back();
    }

    static char *input;
    static size_t __len;
    static size_t *ptr;

    void __error(const char *expected, char encountered, size_t index) {
        std::stringstream stream;
        stream << "expected " << expected << " at index " << index << ", but encountered '" << encountered << "'";
        stream << std::endl;
        input[index + 20] = 0;
        for (int i = index - 20; i < index + 20; ++i)
            if (input[i] == 0) input[i] = ' ';
        stream << "context: " << (input + index - 20) << std::endl;
        stream << "         " << std::string(20, ' ') << "^";
        throw std::runtime_error(stream.str());
    }

#define next_char() ({ \
        idx = *ptr++; \
        if (idx >= __len) throw std::runtime_error("text ended prematurely"); \
        ch = input[idx]; \
    })

#define peek_char() ({ \
        idx = *ptr; \
        ch = input[idx]; \
    })

#define expect(__char) ({ \
        if (ch != (__char)) __error(#__char, ch, idx); \
    })

#define error(__expected) ({ \
        __error(__expected, ch, idx); \
    })

    JsonValue _parseValue();

    JsonValue _parseObject() {
        size_t idx;
        char ch;
        peek_char();
        auto *object = new JsonObject;
        if (ch == '}') {
            next_char();
            return JsonValue(object);
        }
        while (true) {
            expect('"');
            std::string key = parseStr(input + idx);  // keys are probably short strings
            next_char();
            next_char();
            expect(':');
            JsonValue value = _parseValue();
            (*object)[key] = value;
            next_char();
            if (ch == '}') break;
            expect(',');
            peek_char();
        }
        return JsonValue(object);
    }

    JsonValue _parseArray() {
        size_t idx;
        char ch;
        peek_char();
        auto *array = new JsonArray;
        if (ch == ']') {
            next_char();
            return JsonValue(array);
        }
        while (true) {
            JsonValue value = _parseValue();
            array->push_back(value);
            next_char();
            if (ch == ']') break;
            expect(',');
        }
        return JsonValue(array);
    }

    JsonValue _parseValue() {
        size_t idx;
        char ch;
        next_char();
        JsonValue value;
        switch (ch) {
            case '"':
#if USE_PARSE_STR_AVX
                value = JsonValue(parseStrAVX(input + idx));
#else
                value = JsonValue(parseStr(input + idx));
#endif
                break;
            case 't':
                value = JsonValue(parseTrue(input + idx));
                break;
            case 'f':
                value = JsonValue(parseFalse(input + idx));
                break;
            case 'n':
                parseNull(input + idx);
                value = JsonValue();
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
                value = parseNumber(input + idx);
                break;
            case '[':
                value = _parseArray();
                break;
            case '{':
                value = _parseObject();
                break;
            default:
                error("JSON value");
        }
        return value;
    }

    JsonValue parseJson(char *document, size_t size) {
        input = document;
        __len = size;

        size_t base = 0;
        size_t *indices = new size_t[size / 2 + 2];  // TODO: Make this a dynamic-sized array

        u_int64_t prev_escape_mask = 0;
        u_int64_t prev_quote_mask = 0;
        u_int64_t prev_pseudo_mask = 0;
        for (size_t offset = 0; offset < size; offset += 64) {
            __m256i _input1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(document + offset));
            __m256i _input2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(document + offset + 32));
            MercuryJson::Warp input(_input2, _input1);
            u_int64_t escape_mask = MercuryJson::extract_escape_mask(input, &prev_escape_mask);
            u_int64_t quote_mask = 0;
            u_int64_t literal_mask = MercuryJson::extract_literal_mask(
                    input, escape_mask, &prev_quote_mask, &quote_mask);
            u_int64_t structural_mask = 0, whitespace_mask = 0;
            MercuryJson::extract_structural_whitespace_characters(
                    input, literal_mask, &structural_mask, &whitespace_mask);
            u_int64_t pseudo_mask = MercuryJson::extract_pseudo_structural_mask(
                    structural_mask, whitespace_mask, quote_mask, literal_mask, &prev_pseudo_mask);
            MercuryJson::construct_structural_character_pointers(pseudo_mask, offset, indices, &base);
        }

        ptr = indices;

        auto value = _parseValue();
        delete[] indices;

        return value;
    }


//    void parse(char *input, size_t len, size_t *indices) {
//        std::deque<JsonValue> values(MAX_DEPTH);
//        std::vector<int> state_stack(MAX_DEPTH);
//
//        size_t idx;
//        ParserState state = START;
//        std::string key;
//        JsonObject *cur_obj;
//        JsonArray *cur_arr;
//        JsonValue value;
//        while ((idx = *indices++) < len) {
//            JsonValue *last = get_last(values);
//            char ch = input[idx];
//
//            switch (state) {
//                case START:
//                    switch (ch) {
//                        case '{':
//                            state_stack.push_back(state);
//                            state = OBJECT_BEGIN;
//                            break;
//                        case '[':
//                            state_stack.push_back(state);
//                            state = ARRAY_BEGIN;
//                            break;
//                        default:
//                            error("'{' or '['");
//                    }
//                    break;
//                case OBJECT_BEGIN:
//                    switch (ch) {
//                        case '}':
//                            next_state(SCOPE_END);
//                            break;
//                        case '"':
//                            key = parseStr(input + idx + 1);
//                            state = OBJECT_ELEMS;
//                            break;
//                        default:
//                            error("'}' or '\"'");
//                    }
//                case OBJECT_ELEMS:
//                    expect_next(':');
//                    cur_obj = values.back().object;
//                    switch (ch) {
//                        case '"':
//                            value = JsonValue(parseStr(input + idx + 1));
//                            break;
//                        case 't':
//                            value = JsonValue(parseTrue(input + idx));
//                            break;
//                        case 'f':
//                            value = JsonValue(parseFalse(input + idx));
//                            break;
//                        case 'n':
//                            parseNull(input + idx);
//                            value = JsonValue();
//                            break;
//                        case '0':
//                        case '1':
//                        case '2':
//                        case '3':
//                        case '4':
//                        case '5':
//                        case '6':
//                        case '7':
//                        case '8':
//                        case '9':
//                        case '-':
//                            value = parseNumeric(input + idx);
//                            break;
//                        case '[':
//                            state_stack.push_back(state);
//                            state =
//                    }
//                    break;
//                case ARRAY_ELEMS:
//                    break;
//                case OBJECT_CONTINUE:
//                    break;
//                case ARRAY_BEGIN:
//                    break;
//                case SCOPE_END:
//                    break;
//                case ARRAY_ELEMS:
//                    break;
//            }
//            switch (ch) {
//                case '{':
//                    if (last && (last->type == JsonValue::TYPE_CHAR))
//                        throw std::runtime_error("found '{' after non-structural character");
//                    values.push_back(JsonValue('{'));
//                    break;
//                case '[':
//                    if (last && (last->type == JsonValue::TYPE_CHAR))
//                        throw std::runtime_error("found '[' after non-structural character");
//                    values.push_back(JsonValue('['));
//                    break;
//                case ']':
//
//                    break;
//                case '}':
//                    break;
//                case ':':
//                    if (last && last->type != JsonValue::TYPE_STR)
//                        throw std::runtime_error("found ':' after non-string value");
//                    values.push_back(JsonValue(':'));
//                    break;
//                case ',':
//                    if (last && (last->type == JsonValue::TYPE_CHAR))
//                        throw std::runtime_error("found ',' after structural character");
//                    values.push_back(JsonValue(','));
//                    break;
//                case '"':
//                    if (last && (last->type != JsonValue::TYPE_CHAR))
//                        throw std::runtime_error("found string after non-structural character");
//                    // values.push_back(JsonValue(parseStr(input + idx + 1)));
//                    break;
//                case '0':
//                case '1':
//                case '2':
//                case '3':
//                case '4':
//                case '5':
//                case '6':
//                case '7':
//                case '8':
//                case '9':
//                case '-':
//                    if (last && (last->type != JsonValue::TYPE_CHAR))
//                        throw std::runtime_error("found number after non-structural character");
//                    // stack.push(JsonValue(parseStr(input + idx + 1)));
//                    break;
//                case 'n':
//                    if (last && (last->type != JsonValue::TYPE_CHAR))
//                        throw std::runtime_error("found null after non-structural character");
//                    parseNull(input + idx);
//                    values.push_back(JsonValue());
//                    break;
//                case 'f':
//                    if (last && (last->type != JsonValue::TYPE_CHAR))
//                        throw std::runtime_error("found true/false after non-structural character");
//                    values.push_back(JsonValue(parseFalse(input + idx)));
//                    break;
//                case 't':
//                    if (last && (last->type != JsonValue::TYPE_CHAR))
//                        throw std::runtime_error("found true/false after non-structural character");
//                    values.push_back(JsonValue(parseTrue(input + idx)));
//                    break;
//                default:
//                    break;
//            }
//        }
//    }

    char *parseStrAVX(char *s) {
        ++s;  // skip the " character
        u_int64_t prev_odd_backslash_ending_mask = 0ULL;
        char *dest = s, *base = s;
        while (true) {
            Warp input(s);
            u_int64_t escape_mask = extract_escape_mask<true>(input, &prev_odd_backslash_ending_mask);
            u_int64_t quote_mask = __cmpeq_mask<'"'>(input) & (~escape_mask);

            size_t ending_offset = _tzcnt_u64(quote_mask);
#if !PARSE_STR_FULL_AVX
            /* old version */
            size_t last_offset = 0, length;
            while (true) {
                size_t offset = _tzcnt_u64(escape_mask);
                length = offset - last_offset;
                char escaper = s[offset];
                memmove(dest, s + last_offset, length);
                dest += length;
                if (offset >= ending_offset) break;
                *(dest - 1) = escape_map[escaper];
                last_offset = offset + 1;
                escape_mask = _blsr_u64(escape_mask);
            }
            s += 64;
            if (ending_offset < 64) {
                dest[ending_offset - last_offset - length] = '\0';
                break;
            }
            // similar bug here
#else
            /* new version */
            __m256i lo_mask = convert_to_mask(escape_mask);
            __m256i hi_mask = convert_to_mask(escape_mask >> 32U);
            // mask ? translated : original
            __m256i lo_trans = translate_escape_characters(input.lo);
            __m256i hi_trans = translate_escape_characters(input.hi);
            input.lo = _mm256_blendv_epi8(lo_trans, input.lo, lo_mask);
            input.hi = _mm256_blendv_epi8(hi_trans, input.hi, hi_mask);
            u_int64_t escaper_mask = (escape_mask >> 1U) | (prev_odd_backslash_ending_mask << 63U);

            if (ending_offset < 64) {
                u_int64_t str_mask = (1ULL << (ending_offset / 2) << ((ending_offset + 1) / 2)) - 1ULL;
                escaper_mask &= str_mask;
                deescape(input, escaper_mask);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest), input.lo);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest + 32), input.hi);
                size_t end_pos = _mm_popcnt_u64(str_mask & ~escaper_mask);
                memmove(s + ending_offset, dest + end_pos, 64 - ending_offset);
                dest[end_pos] = '\0';
                break;
            } else {
                deescape(input, escaper_mask);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest), input.lo);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest + 32), input.hi);
                dest += 64 - _mm_popcnt_u64(escaper_mask);
                s += 64;
            }
#endif
        }
        return base;
    }

    inline __m256i convert_to_mask(u_int32_t input) {
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

    inline u_int64_t __extract_highestbit_pext(const Warp &input, int shift, u_int64_t escaper_mask) {
        u_int64_t lo = static_cast<u_int32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(input.lo, shift)));
        u_int64_t hi = static_cast<u_int32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(input.hi, shift)));
        return _pext_u64(((hi << 32U) | lo), escaper_mask);
    }

    inline __m256i __expand(u_int32_t input) {
        /* 0x00 for 1-bits, 0x01 for 0-bits */
        const __m256i ones = _mm256_set1_epi8(1);
        return _mm256_add_epi8(convert_to_mask(input), ones);
    }

    inline __m256i __reconstruct(u_int32_t h0, u_int32_t h1, u_int32_t h2, u_int32_t h3,
                                 u_int32_t h4, u_int32_t h5, u_int32_t h6, u_int32_t h7) {
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

    void deescape(Warp &input, u_int64_t escaper_mask) {
        /* Remove 8-bit characters specified by `escaper_mask` */
        u_int64_t nonescaper_mask = ~escaper_mask;
        // Obtain each bit from each 8-bit character, keeping only non-escapers
        u_int64_t h7 = __extract_highestbit_pext(input, 0, nonescaper_mask);
        u_int64_t h6 = __extract_highestbit_pext(input, 1, nonescaper_mask);
        u_int64_t h5 = __extract_highestbit_pext(input, 2, nonescaper_mask);
        u_int64_t h4 = __extract_highestbit_pext(input, 3, nonescaper_mask);
        u_int64_t h3 = __extract_highestbit_pext(input, 4, nonescaper_mask);
        u_int64_t h2 = __extract_highestbit_pext(input, 5, nonescaper_mask);
        u_int64_t h1 = __extract_highestbit_pext(input, 6, nonescaper_mask);
        u_int64_t h0 = __extract_highestbit_pext(input, 7, nonescaper_mask);
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
