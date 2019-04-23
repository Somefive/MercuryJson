#include <cmath>
#include <cstdio>
#include <cstring>
#include <immintrin.h>

#include <algorithm>
#include <bitset>
#include <deque>
#include <map>
#include <sstream>
#include <stack>
#include <string>
#include <variant>
#include <vector>

#include "mercuryparser.h"

#ifndef STATIC_CMPEQ_MASK
#define STATIC_CMPEQ_MASK 0
#endif

#ifndef PARSE_STR_MODE
#define PARSE_STR_MODE 1  // 0 for naive, 1 for avx, 2 for per_bit
#endif

#ifndef PARSE_STR_FULLY_AXV
#define PARSE_STR_FULLY_AXV 0
#endif

#ifndef PARSE_NUMBER_AVX
#define PARSE_NUMBER_AVX 1
#endif

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
    u_int64_t extract_escape_mask(const Warp &raw, u_int64_t *prev_odd_backslash_ending_mask) {
        u_int64_t backslash_mask = __cmpeq_mask<'\\'>(raw);
        u_int64_t start_backslash_mask = backslash_mask & ~(backslash_mask << 1U);
        u_int64_t even_start_backslash_mask = (start_backslash_mask & __even_mask64) ^ *prev_odd_backslash_ending_mask;
        u_int64_t even_carrier_backslash_mask = even_start_backslash_mask + backslash_mask;
        u_int64_t even_escape_mask;
        even_escape_mask = (even_carrier_backslash_mask ^ backslash_mask) & __odd_mask64;

        u_int64_t odd_start_backslash_mask = (start_backslash_mask & __odd_mask64) ^ *prev_odd_backslash_ending_mask;
        u_int64_t odd_carrier_backslash_mask = odd_start_backslash_mask + backslash_mask;
        u_int64_t odd_backslash_ending_mask = odd_carrier_backslash_mask < odd_start_backslash_mask;
        *prev_odd_backslash_ending_mask = odd_backslash_ending_mask;
        u_int64_t odd_escape_mask;
        odd_escape_mask = (odd_carrier_backslash_mask ^ backslash_mask) & __even_mask64;
        return even_escape_mask | odd_escape_mask;
    }
    // @formatter:on

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

    inline void __error_maybe_escape(char *context, size_t *length, char ch) {
        if (ch == '\t' || ch == '\n' || ch == '\b' || ch == '\0') {
            context[*length++] = '\\';
            context[*length++] = '[';
            switch (ch) {
                case '\t':
                    context[*length++] = 't';
                    break;
                case '\n':
                    context[*length++] = 'n';
                    break;
                case '\b':
                    context[*length++] = 'b';
                    break;
                case '\0':
                    context[*length++] = '0';
                    break;
            }
            context[*length++] = ']';
        } else {
            context[*length++] = ch;
        }
    }

    [[noreturn]] void __error(const std::string &message, const char *input, size_t offset) {
        static const size_t context_len = 20;
        char *context = new char[(2 * context_len + 1) * 2];  // add space for escaped chars
        size_t length = 0U;
        for (size_t i = offset > context_len ? offset - context_len : 0U; i < offset; ++i)
            __error_maybe_escape(context, &length, input[i]);
        size_t left = length;
        for (size_t i = offset; i < offset + context_len; ++i) {
            if (input[i] == '\0') break;
            __error_maybe_escape(context, &length, input[i]);
        }
        std::stringstream stream;
        stream << message << std::endl;
        stream << "context: " << context << std::endl;
        delete[] context;
        stream << "         " << std::string(left, ' ') << "^";
        throw std::runtime_error(stream.str());
    }

    inline bool _all_digits(const char *s) {
        u_int64_t val = *reinterpret_cast<const u_int64_t *>(s);
        return (((val & 0xf0f0f0f0f0f0f0f0)
                 | (((val + 0x0606060606060606) & 0xf0f0f0f0f0f0f0f0) >> 4U)) == 0x3333333333333333);
    }

    inline u_int32_t _parse_eight_digits(const char *s) {
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

    char *parse_str_naive(char *s, size_t *len, size_t offset) {
        bool escape = false;
        char *ptr = s + offset;
        for (char *end = s + offset; escape || *end != '"'; ++end) {
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
                        __error("invalid escape sequence", s, end - s);
                }
                escape = false;
            } else {
                if (*end == '\\') escape = true;
                else *ptr++ = *end;
            }
        }
        *ptr++ = 0;
        if (len != nullptr) *len = ptr - s;
        return s;
    }

    bool parse_true(const char *s, size_t offset) {
        const auto *literal = reinterpret_cast<const u_int32_t *>(s + offset);
        u_int32_t target = 0x65757274;
        if (target != *literal || !structural_or_whitespace[s[offset + 4]])
            __error("invalid true value", s, offset);
        return true;
    }

    bool parse_false(const char *s, size_t offset) {
        const auto *literal = reinterpret_cast<const u_int64_t *>(s + offset);
        u_int64_t target = 0x00000065736c6166;
        u_int64_t mask = 0x000000ffffffffff;
        if (target != (*literal & mask) || !structural_or_whitespace[s[offset + 5]])
            __error("invalid false value", s, offset);
        return false;
    }

    void parse_null(const char *s, size_t offset) {
        const auto *literal = reinterpret_cast<const u_int32_t *>(s + offset);
        u_int32_t target = 0x6c6c756e;
        if (target != *literal || !structural_or_whitespace[s[offset + 4]])
            __error("invalid null value", s, offset);
    }

    void JSON::_error(const char *expected, char encountered, size_t index) {
        std::stringstream stream;
        stream << "expected " << expected << " at index " << index << ", but encountered '" << encountered << "'";
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
#if PARSE_STR_MODE == 2
        size_t len = *idx_ptr - idx;
        char *dest = allocator.allocate<char>(len, 32);
        parse_str_per_bit(input + idx + 1, dest);
        return dest;
#elif PARSE_STR_MODE == 1
        return parse_str_avx(input + idx + 1);
#else
        return parse_str_naive(input + idx + 1);
#endif
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

    JSON::JSON(char *document, size_t size, bool manual_construct) : allocator(
#if PARSE_STR_MODE == 2
            size * 2  // when using parse_str_per_bit, we allocate memory for parsed strings
#else
            size
#endif
    ) {
        this->input = document;
        this->input_len = size;
        this->document = nullptr;

        this->idx_ptr = this->indices = new size_t[size];  // TODO: Make this a dynamic-sized array

        if (!manual_construct) {
            exec_stage1();
            exec_stage2();
        }
    }

    void JSON::exec_stage1() {
        size_t base = 0;
        u_int64_t prev_escape_mask = 0;
        u_int64_t prev_quote_mask = 0;
        u_int64_t prev_pseudo_mask = 0;
        for (size_t offset = 0; offset < this->input_len; offset += 64) {
            __m256i _input1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(this->input + offset));
            __m256i _input2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(this->input + offset + 32));
            Warp warp(_input2, _input1);
            u_int64_t escape_mask = extract_escape_mask(warp, &prev_escape_mask);
            u_int64_t quote_mask = 0;
            u_int64_t literal_mask = extract_literal_mask(warp, escape_mask, &prev_quote_mask, &quote_mask);
            u_int64_t structural_mask = 0, whitespace_mask = 0;
            extract_structural_whitespace_characters(warp, literal_mask, &structural_mask, &whitespace_mask);
            u_int64_t pseudo_mask = extract_pseudo_structural_mask(
                    structural_mask, whitespace_mask, quote_mask, literal_mask, &prev_pseudo_mask);
            construct_structural_character_pointers(pseudo_mask, offset, this->indices, &base);
        }
    }

    void JSON::exec_stage2() {
        this->document = _parse_value();
        delete[] this->indices;
        this->indices = nullptr;
    }

    JSON::~JSON() = default;

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
//                            key = parse_str_naive(input + idx + 1);
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
//                            value = JsonValue(parse_str_naive(input + idx + 1));
//                            break;
//                        case 't':
//                            value = JsonValue(parse_true(input + idx));
//                            break;
//                        case 'f':
//                            value = JsonValue(parse_false(input + idx));
//                            break;
//                        case 'n':
//                            parse_null(input + idx);
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
//                    // values.push_back(JsonValue(parse_str_naive(input + idx + 1)));
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
//                    // stack.push(JsonValue(parse_str_naive(input + idx + 1)));
//                    break;
//                case 'n':
//                    if (last && (last->type != JsonValue::TYPE_CHAR))
//                        throw std::runtime_error("found null after non-structural character");
//                    parse_null(input + idx);
//                    values.push_back(JsonValue());
//                    break;
//                case 'f':
//                    if (last && (last->type != JsonValue::TYPE_CHAR))
//                        throw std::runtime_error("found true/false after non-structural character");
//                    values.push_back(JsonValue(parse_false(input + idx)));
//                    break;
//                case 't':
//                    if (last && (last->type != JsonValue::TYPE_CHAR))
//                        throw std::runtime_error("found true/false after non-structural character");
//                    values.push_back(JsonValue(parse_true(input + idx)));
//                    break;
//                default:
//                    break;
//            }
//        }
//    }

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
                    u_int8_t escaped = escape_map[escape_char];
                    if (escaped == 0U)
                        __error("invalid escape character '" + std::string(1, escape_char) + "'", _src, offset);
                    dest[backslash_offset] = escape_map[escape_char];
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

    char *parse_str_avx(char *src, size_t *len, size_t offset) {
        char *_src = src;
        src += offset;
        char *dest = src, *base = src;
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
                        dest[ending_offset - last_offset] = '\0';
                        if (len != nullptr) *len = (dest - base) + (ending_offset - last_offset);
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
#if PARSE_STR_FULLY_AXV
                /* fully-AVX version */
                __m256i lo_mask = convert_to_mask(escape_mask);
                __m256i hi_mask = convert_to_mask(escape_mask >> 32U);
                // mask ? translated : original
                __m256i lo_trans = translate_escape_characters(input.lo);
                __m256i hi_trans = translate_escape_characters(input.hi);
                input.lo = _mm256_blendv_epi8(lo_trans, input.lo, lo_mask);
                input.hi = _mm256_blendv_epi8(hi_trans, input.hi, hi_mask);
                u_int64_t escaper_mask = (escape_mask >> 1U) | (prev_odd_backslash_ending_mask << 63U);

                deescape(input, escaper_mask);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest), input.lo);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest + 32), input.hi);
                dest += 64 - _mm_popcnt_u64(escaper_mask);
                src += 64;
#else
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
#endif
            }
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
