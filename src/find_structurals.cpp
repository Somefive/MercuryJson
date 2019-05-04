#include "mercuryparser.h"

#include <immintrin.h>
#include <stdio.h>


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
        uint64_t backslash_mask = __cmpeq_mask(raw, '\\');
        uint64_t start_backslash_mask = backslash_mask & ~(backslash_mask << 1U);

        uint64_t even_start_backslash_mask = (start_backslash_mask & kEvenMask64) ^ *prev_odd_backslash_ending_mask;
        uint64_t even_carrier_backslash_mask = even_start_backslash_mask + backslash_mask;
        uint64_t even_escape_mask = (even_carrier_backslash_mask ^ backslash_mask) & kOddMask64;

        uint64_t odd_start_backslash_mask = (start_backslash_mask & kOddMask64) ^ *prev_odd_backslash_ending_mask;
        uint64_t odd_carrier_backslash_mask = odd_start_backslash_mask + backslash_mask;
        uint64_t odd_escape_mask = (odd_carrier_backslash_mask ^ backslash_mask) & kEvenMask64;

        uint64_t odd_backslash_ending_mask = odd_carrier_backslash_mask < odd_start_backslash_mask;
        *prev_odd_backslash_ending_mask = odd_backslash_ending_mask;

        return even_escape_mask | odd_escape_mask;
    }
    // @formatter:on

    uint64_t extract_literal_mask(
            const Warp &raw, uint64_t escape_mask, uint64_t *prev_literal_ending, uint64_t *quote_mask) {
        uint64_t _quote_mask = __cmpeq_mask(raw, '"') & ~escape_mask;
        uint64_t literal_mask = _mm_cvtsi128_si64(
                _mm_clmulepi64_si128(_mm_set_epi64x(0ULL, _quote_mask), _mm_set1_epi8(0xFF), 0));
        literal_mask ^= *prev_literal_ending;
        *quote_mask = _quote_mask;
        *prev_literal_ending = static_cast<uint64_t>(static_cast<int64_t>(literal_mask) >> 63);
        return literal_mask;
    }

    void extract_structural_whitespace_characters(
            const Warp &raw, uint64_t literal_mask, uint64_t *structural_mask, uint64_t *whitespace_mask) {
        const __m256i upper_lookup = _mm256_setr_epi8(8, 0, 17, 2, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0,
                                                      8, 0, 17, 2, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0);
        const __m256i lower_lookup = _mm256_setr_epi8(16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 4, 1, 12, 0, 0,
                                                      16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 4, 1, 12, 0, 0);

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
        *whitespace_mask = ~(__cmpeq_mask(hi_whitespace_mask, lo_whitespace_mask, 0) | literal_mask);

        __m256i hi_structural_mask = _mm256_and_si256(hi_character_label, _mm256_set1_epi8(0x7));
        __m256i lo_structural_mask = _mm256_and_si256(lo_character_label, _mm256_set1_epi8(0x7));
        *structural_mask = ~(__cmpeq_mask(hi_structural_mask, lo_structural_mask, 0) | literal_mask);
    }

    uint64_t extract_pseudo_structural_mask(
            uint64_t structural_mask, uint64_t whitespace_mask, uint64_t quote_mask, uint64_t literal_mask,
            uint64_t *prev_pseudo_structural_end_mask) {
        uint64_t st_ws = structural_mask | whitespace_mask;
        structural_mask |= quote_mask;
        uint64_t pseudo_structural_mask = (st_ws << 1U) | *prev_pseudo_structural_end_mask;
        *prev_pseudo_structural_end_mask = st_ws >> 63U;
        pseudo_structural_mask &= (~whitespace_mask) & (~literal_mask);
        structural_mask |= pseudo_structural_mask;
        structural_mask &= ~(quote_mask & ~literal_mask);
        return structural_mask;
    }

    const size_t kStructuralUnrollCount = 8;

    void construct_structural_character_pointers(
            uint64_t pseudo_structural_mask, size_t offset, size_t *indices, size_t *base) {
        size_t next_base = *base + __builtin_popcountll(pseudo_structural_mask);
        while (pseudo_structural_mask) {
            for (size_t i = 0; i < kStructuralUnrollCount; ++i) {
                indices[*base + i] = offset + _tzcnt_u64(pseudo_structural_mask);
                pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            }
            *base += kStructuralUnrollCount;
        }
        *base = next_base;
    }

    void __printChar_m256i(__m256i raw) {
        auto *vals = reinterpret_cast<uint8_t *>(&raw);
        for (size_t i = 0; i < 32; ++i) printf("%2x(%c) ", vals[i], vals[i]);
        printf("\n");
    }
}
