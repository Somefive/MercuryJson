#include <cstdio>
#include <cstring>
#include <immintrin.h>

#include <deque>
#include <map>
#include <stack>
#include <string>
#include <vector>

#include "mercuryparser.h"

namespace MercuryJson {

    inline u_int64_t __cmpeq_mask(const Warp &raw, char c) {
        u_int64_t hi = static_cast<u_int32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(raw.hi, _mm256_set1_epi8(c))));
        u_int64_t lo = static_cast<u_int32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(raw.lo, _mm256_set1_epi8(c))));
        // printf("__cmpeq_mask, hi: %16llx, lo: %16llx\n", (unsigned long long)(hi << 32), (unsigned long long)(lo));
        return (hi << 32U) | lo;
    }

    void __print(Warp &raw) {
        auto *vals = reinterpret_cast<u_int8_t *>(&raw);
        for (size_t i = 0; i < 64; ++i) printf("%2x ", vals[i]);
        printf("\n");
    }

    const u_int64_t __even_mask64 = 0x5555555555555555U;
    const u_int64_t __odd_mask64 = ~__even_mask64;

    u_int64_t extract_escape_mask(const Warp &raw, u_int64_t &prev_odd_backslash_ending_mask) {
        u_int64_t backslash_mask = __cmpeq_mask(raw, '\\');
        u_int64_t start_backslash_mask = backslash_mask & ~(backslash_mask << 1U);
        u_int64_t even_start_backslash_mask = (start_backslash_mask & __even_mask64) ^prev_odd_backslash_ending_mask;
        u_int64_t even_carrier_backslash_mask = even_start_backslash_mask + backslash_mask;
        u_int64_t even_escape_mask = even_carrier_backslash_mask & (~backslash_mask) & __odd_mask64;
        u_int64_t odd_start_backslash_mask = (start_backslash_mask & __odd_mask64) ^prev_odd_backslash_ending_mask;
        u_int64_t odd_carrier_backslash_mask = odd_start_backslash_mask + backslash_mask;
        u_int64_t odd_backslash_ending_mask =
                (odd_carrier_backslash_mask < odd_start_backslash_mask) | prev_odd_backslash_ending_mask;
        prev_odd_backslash_ending_mask = odd_backslash_ending_mask;
        u_int64_t odd_escape_mask = odd_carrier_backslash_mask & (~backslash_mask) & __even_mask64;
        // printf("backslash_mask: %016llx\n", static_cast<unsigned long long>(backslash_mask));
        // printf("start_backslash_mask: %016llx\n", static_cast<unsigned long long>(start_backslash_mask));
        // printf("even_start_backslash_mask: %016llx\n", static_cast<unsigned long long>(even_start_backslash_mask));
        // printf("even_carrier_backslash_mask: %016llx\n", static_cast<unsigned long long>(even_carrier_backslash_mask));
        // printf("even_escape_mask: %016llx\n", static_cast<unsigned long long>(even_escape_mask));
        // printf("odd_start_backslash_mask: %016llx\n", static_cast<unsigned long long>(odd_start_backslash_mask));
        // printf("odd_carrier_backslash_mask: %016llx\n", static_cast<unsigned long long>(odd_carrier_backslash_mask));
        // printf("odd_backslash_ending_mask: %016llx\n", static_cast<unsigned long long>(odd_backslash_ending_mask));
        // printf("odd_backslash_ending_mask: %016llx\n", static_cast<unsigned long long>(odd_backslash_ending_mask));
        return even_escape_mask | odd_escape_mask;
    }

    u_int64_t extract_literal_mask(
            const Warp &raw, u_int64_t escape_mask, u_int64_t &prev_literal_ending, u_int64_t &quote_mask) {
        quote_mask = __cmpeq_mask(raw, '"') & ~escape_mask;
        u_int64_t literal_mask = _mm_cvtsi128_si64(
                _mm_clmulepi64_si128(_mm_set_epi64x(0ULL, quote_mask), _mm_set1_epi8(0xFF), 0));
        u_int64_t literal_reversor = prev_literal_ending * ~0ULL;
        literal_mask ^= literal_reversor;
        prev_literal_ending = (literal_mask & (1ULL << 63U)) >> 63U;
        return literal_mask;
    }

    void extract_structural_whitespace_characters(
            const Warp &raw, u_int64_t literal_mask, u_int64_t &structural_mask, u_int64_t &whitespace_mask) {
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
        whitespace_mask = (~__cmpeq_mask(Warp(hi_whitespace_mask, lo_whitespace_mask), 0) & (~literal_mask));

        __m256i hi_structural_mask = _mm256_and_si256(hi_character_label, _mm256_set1_epi8(0x7));
        __m256i lo_structural_mask = _mm256_and_si256(lo_character_label, _mm256_set1_epi8(0x7));
        structural_mask = (~__cmpeq_mask(Warp(hi_structural_mask, lo_structural_mask), 0) & (~literal_mask));
    }

    u_int64_t extract_pseudo_structural_mask(
            u_int64_t structural_mask, u_int64_t whitespace_mask, u_int64_t quote_mask, u_int64_t literal_mask,
            u_int64_t &prev_pseudo_structural_end_mask) {
        u_int64_t st_ws = structural_mask | whitespace_mask;
        structural_mask |= quote_mask;
        u_int64_t pseudo_structural_mask = (st_ws << 1U) | prev_pseudo_structural_end_mask;
        prev_pseudo_structural_end_mask = (st_ws >> 63U) & 1ULL;
        pseudo_structural_mask &= (~whitespace_mask) & (~literal_mask);
        structural_mask |= pseudo_structural_mask;
        structural_mask &= ~(quote_mask & ~literal_mask);
        return structural_mask;
    }

    void construct_structural_character_pointers(
            u_int64_t pseudo_structural_mask, size_t offset, size_t *indices, size_t &base) {
        size_t next_base = base + __builtin_popcountll(pseudo_structural_mask);
        while (pseudo_structural_mask) {
            indices[base] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[base + 1] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[base + 2] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[base + 3] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[base + 4] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[base + 5] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[base + 6] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            indices[base + 7] = offset + _tzcnt_u64(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            base += 8;
        }
        base = next_base;
    }

    const __mmask32 __even_mask = 0x55555555U;
    const __mmask32 __odd_mask = ~__even_mask;

    inline __mmask32 __cmpeq_mask(__m256i raw, char c) {
        return static_cast<__mmask32>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(raw, _mm256_set1_epi8(c))));
    }

    void __print_m256i(__m256i raw) {
        u_int8_t *vals = reinterpret_cast<u_int8_t *>(&raw);
        for (size_t i = 0; i < 32; ++i) printf("%2x ", vals[i]);
        printf("\n");
    }

    __mmask32 extract_escape_mask(__m256i raw, __mmask32 &prev_odd_backslash_ending_mask) {
        __mmask32 backslash_mask = __cmpeq_mask(raw, '\\');
        // printf("backslash_mask: %8x\n", backslash_mask);
        __mmask32 start_backslash_mask = backslash_mask & ~(backslash_mask << 1U);
        // printf("start_backslash_mask: %8x\n", start_backslash_mask);
        __mmask32 even_start_backslash_mask = (start_backslash_mask & __even_mask) ^prev_odd_backslash_ending_mask;
        // printf("even_start_backslash_mask: %8x\n", even_start_backslash_mask);
        __mmask32 even_carrier_backslash_mask = even_start_backslash_mask + backslash_mask;
        // printf("even_carrier_backslash_mask: %8x\n", even_carrier_backslash_mask);
        __mmask32 even_escape_mask = even_carrier_backslash_mask & (~backslash_mask) & __odd_mask;
        // printf("even_escape_mask: %8x\n", even_escape_mask);

        __mmask32 odd_start_backslash_mask = (start_backslash_mask & __odd_mask) ^prev_odd_backslash_ending_mask;
        // printf("odd_start_backslash_mask: %8x\n", odd_start_backslash_mask);
        __mmask32 odd_carrier_backslash_mask = odd_start_backslash_mask + backslash_mask;
        // printf("odd_carrier_backslash_mask: %8x\n", odd_carrier_backslash_mask);
        __mmask32 odd_backslash_ending_mask = odd_carrier_backslash_mask < odd_start_backslash_mask;
        odd_backslash_ending_mask |= prev_odd_backslash_ending_mask;
        prev_odd_backslash_ending_mask = odd_backslash_ending_mask;
        __mmask32 odd_escape_mask = odd_carrier_backslash_mask & (~backslash_mask) & __even_mask;
        // printf("odd_escape_mask: %8x\n", odd_escape_mask);

        return even_escape_mask | odd_escape_mask;
    }

    __mmask32 extract_literal_mask(
            __m256i raw, __mmask32 escape_mask, __mmask32 &prev_literal_ending, __mmask32 &quote_mask) {
        quote_mask = __cmpeq_mask(raw, '"') & ~escape_mask;
        __mmask32 literal_mask = _mm_cvtsi128_si32(
                _mm_clmulepi64_si128(_mm_set_epi32(0, 0, 0, quote_mask), _mm_set1_epi8(0xFF), 0));
        // printf("literal_mask: %08x\n", literal_mask);
        __mmask32 literal_reversor = prev_literal_ending * ~0U;
        // printf("literal reversor: %08x\n", literal_reversor);
        literal_mask ^= literal_reversor;
        prev_literal_ending = (literal_mask & (1U << (32U - 1))) >> (32U - 1);
        // printf("literal mask: %08x, literal_ending: %08x\n", literal_mask, prev_literal_ending);
        return literal_mask;
    }

    void extract_structural_whitespace_characters(
            __m256i raw, __mmask32 literal_mask, __mmask32 &structural_mask, __mmask32 &whitespace_mask) {
        const __m256i hi_lookup = _mm256_setr_epi8(8, 0, 17, 2, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 17, 2, 0, 4,
                                                   0, 4, 0, 0, 0, 0, 0, 0, 0, 0);
        const __m256i lo_lookup = _mm256_setr_epi8(16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 4, 1, 12, 0, 0, 16, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 8, 10, 4, 1, 12, 0, 0);
        __m256i hi_index = _mm256_shuffle_epi8(hi_lookup,
                                               _mm256_and_si256(_mm256_srli_epi16(raw, 4), _mm256_set1_epi8(0x7F)));
        __m256i lo_index = _mm256_shuffle_epi8(lo_lookup, raw);
        __m256i character_label = _mm256_and_si256(hi_index, lo_index);
        // printf("raw:");
        // __print_m256i(raw);
        // printf("hi_index:");
        // __print_m256i(hi_index);
        // printf("lo_index:");
        // __print_m256i(lo_index);
        // printf("character_label:");
        // __print_m256i(character_label);
        whitespace_mask =
                (~__cmpeq_mask(_mm256_and_si256(character_label, _mm256_set1_epi8(0x18)), 0)) & (~literal_mask);
        structural_mask =
                (~__cmpeq_mask(_mm256_and_si256(character_label, _mm256_set1_epi8(0x7)), 0)) & (~literal_mask);
    }

    __mmask32 extract_pseudo_structural_mask(
            __mmask32 structural_mask, __mmask32 whitespace_mask, __mmask32 quote_mask, __mmask32 literal_mask,
            __mmask32 &prev_pseudo_structural_end_mask) {
        __mmask32 st_ws = structural_mask | whitespace_mask;
        // printf("st_ws: %08x\n", st_ws);
        structural_mask |= quote_mask;
        __mmask32 pseudo_structural_mask = (st_ws << 1U) | prev_pseudo_structural_end_mask;
        prev_pseudo_structural_end_mask = (st_ws >> (32U - 1)) & 1U;
        pseudo_structural_mask &= (~whitespace_mask) & (~literal_mask);
        structural_mask |= pseudo_structural_mask;
        // printf("right quote: %08x\n", (quote_mask & ~literal_mask));
        structural_mask &= ~(quote_mask & ~literal_mask);
        return structural_mask;
    }

    void construct_structural_character_pointers(
            __mmask32 pseudo_structural_mask, size_t offset, size_t *indices, size_t &base) {
        size_t next_base = base + __builtin_popcount(pseudo_structural_mask);
        while (pseudo_structural_mask) {
            indices[base] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base + 1] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base + 2] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base + 3] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base + 4] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base + 5] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base + 6] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base + 7] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            base += 8;
        }
        base = next_base;
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

    char *parseStr(const char *s, char *&buffer) {
        char *base = buffer;
        u_int64_t prev_odd_backslash_ending_mask = 0;
        while (true) {
            __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s));
            __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + 32));
            Warp warp(lo, hi);
            u_int64_t escape_mask = extract_escape_mask(warp, prev_odd_backslash_ending_mask);
            u_int64_t quote_mask = __cmpeq_mask(warp, '"') & escape_mask;

            u_int64_t bitcnt = 0;
            while (escape_mask) {
                u_int64_t shift = _tzcnt_u64(escape_mask);
                memcpy(buffer, s, shift - bitcnt);

                buffer[shift - 1] = escape_map[s[shift]];
                buffer += shift;
                s += shift + 1;
                bitcnt += shift + 1;
            }
        }
        // TODO: should process escape and termination
        return base;
    }

    bool parseTrue(const char *s) {
        u_int64_t local = 0, mask = 0x0000000065757274;
        std::memcpy(&local, s, 4);
        if (mask != local || !structural_or_whitespace[s[4]]) throw "invalid true value";
        return true;
    }

    bool parseFalse(const char *s) {
        u_int64_t local = 0, mask = 0x00000065736c6166;
        std::memcpy(&local, s, 5);
        if (mask != local || !structural_or_whitespace[s[5]]) throw "invalid false value";
        return false;
    }

    void parseNull(const char *s) {
        u_int64_t local = 0, mask = 0x000000006c6c756e;
        std::memcpy(&local, s, 4);
        if (mask != local || !structural_or_whitespace[s[4]]) throw "invalid null value";
        return;
    }

    JsonValue *get_last(std::deque<JsonValue> &values) {
        return values.empty() ? nullptr : &values.back();
    }

    const JsonValue *get_last(const std::deque<JsonValue> &values) {
        return values.empty() ? nullptr : &values.back();
    }

    void parse(char *input, size_t len, size_t *indices) {
        std::deque<JsonValue> values;
        // std::deque<
        while (true) {
            size_t idx = *indices++;
            if (idx >= len) break;
            JsonValue *last = get_last(values);
            switch (input[idx]) {
                case '{':
                    if (last && (last->type == JsonValue::TYPE_CHAR))
                        throw std::runtime_error("found '{' after non-structural character");
                    values.push_back(JsonValue::create('{'));
                    break;
                case '[':
                    if (last && (last->type == JsonValue::TYPE_CHAR))
                        throw std::runtime_error("found '[' after non-structural character");
                    values.push_back(JsonValue::create('['));
                    break;
                case ']':

                    break;
                case '}':
                    break;
                case ':':
                    if (last && last->type != JsonValue::TYPE_STR)
                        throw std::runtime_error("found ':' after non-string value");
                    values.push_back(JsonValue::create(':'));
                    break;
                case ',':
                    if (last && (last->type == JsonValue::TYPE_CHAR))
                        throw std::runtime_error("found ',' after structural character");
                    values.push_back(JsonValue::create(','));
                    break;
                case '"':
                    if (last && (last->type != JsonValue::TYPE_CHAR))
                        throw std::runtime_error("found string after non-structural character");
                    // values.push_back(JsonValue::create(parseStr(input + idx + 1)));
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
                    if (last && (last->type != JsonValue::TYPE_CHAR))
                        throw std::runtime_error("found number after non-structural character");
                    // stack.push(JsonValue::create(parseStr(input + idx + 1)));
                    break;
                case 'n':
                    if (last && (last->type != JsonValue::TYPE_CHAR))
                        throw std::runtime_error("found null after non-structural character");
                    parseNull(input + idx);
                    values.push_back(JsonValue::create());
                    break;
                case 'f':
                    if (last && (last->type != JsonValue::TYPE_CHAR))
                        throw std::runtime_error("found true/false after non-structural character");
                    values.push_back(JsonValue::create(parseFalse(input + idx)));
                    break;
                case 't':
                    if (last && (last->type != JsonValue::TYPE_CHAR))
                        throw std::runtime_error("found true/false after non-structural character");
                    values.push_back(JsonValue::create(parseTrue(input + idx)));
                    break;
                default:
                    break;
            }
        }
    }

}
