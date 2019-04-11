#include <immintrin.h>
#include <cstdio>
#include "mercuryparser.h"

#define GROUPLEN 32 // =256/8

namespace MercuryJson {

    const __mmask32 __even_mask = 0x55555555U;
    const __mmask32 __odd_mask = ~__even_mask;

    inline __mmask32 __cmpeq_mask(__m256i raw, char c) {
        return static_cast<__mmask32>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(raw, _mm256_set1_epi8(c))));
    }

    void __print_m256i(__m256i raw) {
        u_int8_t * vals = reinterpret_cast<u_int8_t *>(&raw);
        for (size_t i = 0; i < 32; ++i) printf("%2x ", vals[i]);
        printf("\n");
    }

    __mmask32 extract_escape_mask(__m256i raw, __mmask32 & prev_odd_backslash_ending_mask) {
        __mmask32 backslash_mask = __cmpeq_mask(raw, '\\');
        // printf("backslash_mask: %8x\n", backslash_mask);
        __mmask32 start_backslash_mask = backslash_mask & ~(backslash_mask << 1);
        // printf("start_backslash_mask: %8x\n", start_backslash_mask);
        __mmask32 even_start_backslash_mask = start_backslash_mask & __even_mask ^ prev_odd_backslash_ending_mask;
        // printf("even_start_backslash_mask: %8x\n", even_start_backslash_mask);
        __mmask32 even_carrier_backslash_mask = even_start_backslash_mask + backslash_mask;
        // printf("even_carrier_backslash_mask: %8x\n", even_carrier_backslash_mask);
        __mmask32 even_escape_mask = even_carrier_backslash_mask & (~backslash_mask) & __odd_mask;
        // printf("even_escape_mask: %8x\n", even_escape_mask);

        __mmask32 odd_start_backslash_mask = start_backslash_mask & __odd_mask ^ prev_odd_backslash_ending_mask;
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

    __mmask32 extract_literal_mask(__m256i raw, __mmask32 escape_mask, __mmask32 & prev_literal_ending, __mmask32 & quote_mask) {
        quote_mask = __cmpeq_mask(raw, '"') & ~escape_mask;
        __mmask32 literal_mask = _mm_cvtsi128_si32(_mm_clmulepi64_si128(_mm_set_epi32(0, 0, 0, quote_mask), _mm_set1_epi8(0xFF), 0));
        printf("literal_mask: %08x\n", literal_mask);
        __mmask32 literal_reversor = prev_literal_ending * ~0U;
        // printf("literal reversor: %08x\n", literal_reversor);
        literal_mask ^= literal_reversor;
        prev_literal_ending = (literal_mask & (1U << (32 - 1))) >> (32 - 1);
        // printf("literal mask: %08x, literal_ending: %08x\n", literal_mask, prev_literal_ending);
        return literal_mask;
    }

    void extract_structural_whitespace_characters(__m256i raw, __mmask32 literal_mask, __mmask32 & structural_mask, __mmask32 & whitespace_mask) {
        const __m256i hi_lookup = _mm256_setr_epi8(8, 0, 17, 2, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 17, 2, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0);
        const __m256i lo_lookup = _mm256_setr_epi8(16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 4, 1, 12, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 4, 1, 12, 0, 0);
        __m256i hi_index = _mm256_shuffle_epi8(hi_lookup, _mm256_and_si256(_mm256_srli_epi16(raw, 4), _mm256_set1_epi8(0x7F)));
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
        whitespace_mask = (~__cmpeq_mask(_mm256_and_si256(character_label, _mm256_set1_epi8(0x18)), 0)) & (~literal_mask);
        structural_mask = (~__cmpeq_mask(_mm256_and_si256(character_label, _mm256_set1_epi8(0x7)), 0)) & (~literal_mask);
    }

    __mmask32 extract_pseudo_structural_mask(__mmask32 structural_mask, __mmask32 whitespace_mask, __mmask32 quote_mask, __mmask32 literal_mask, __mmask32 & prev_pseudo_structural_end_mask) {
        __mmask32 st_ws = structural_mask | whitespace_mask;
        // printf("st_ws: %08x\n", st_ws);
        structural_mask |= quote_mask;
        __mmask32 pseudo_structural_mask = (st_ws << 1) | prev_pseudo_structural_end_mask;
        prev_pseudo_structural_end_mask = (st_ws >> (32 - 1)) & 1U;
        pseudo_structural_mask &= (~whitespace_mask) & (~literal_mask);
        structural_mask |= pseudo_structural_mask;
        // printf("right quote: %08x\n", (quote_mask & ~literal_mask));
        structural_mask &= ~(quote_mask & ~literal_mask);
        return structural_mask;
    }

    void construct_structural_character_pointers(__mmask32 pseudo_structural_mask, size_t offset, size_t * indices, size_t & base) {
        size_t next_base = base + __builtin_popcount(pseudo_structural_mask);
        while (pseudo_structural_mask) {
            indices[base] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base+1] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base+2] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base+3] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base+4] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base+5] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base+6] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            indices[base+7] = offset + _tzcnt_u32(pseudo_structural_mask);
            pseudo_structural_mask = _blsr_u32(pseudo_structural_mask);
            base += 8;
        }
        base = next_base;
    }

    class ThreadParser {
        
        ThreadParser() {

        }

        void load(const char * inputs, size_t size) {
            const char * end = inputs + size;
            for (const char * begin = inputs; begin != end; begin += GROUPLEN) {
                __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(begin));
                
            }
        }

    };

}