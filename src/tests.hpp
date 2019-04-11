#include <cstdio>
#include "utils.h"
#include "mercuryparser.hpp"
#include <immintrin.h>

void test_extract_mask() {
    size_t size;
    const char * buf = read_file("data/test_extract_escape_mask.json", size);
    printf("%lu: %s\n", size, buf);
    __m256i input1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(buf));
    __m256i input2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(buf+32));
    __mmask32 prev_mask = 0;
    __mmask32 escape_mask1 = MercuryJson::extract_escape_mask(input1, prev_mask);
    __mmask32 escape_mask2 = MercuryJson::extract_escape_mask(input2, prev_mask);
    printf("escape mask: \n%08x\n%08x\n", escape_mask1, escape_mask2);
    prev_mask = 0;
    __mmask32 quote_mask1 = 0, quote_mask2 = 0;
    __mmask32 literal_mask1 = MercuryJson::extract_literal_mask(input1, escape_mask1, prev_mask, quote_mask1);
    __mmask32 literal_mask2 = MercuryJson::extract_literal_mask(input2, escape_mask2, prev_mask, quote_mask2);
    printf("quote mask: \n%08x\n%08x\n", quote_mask1, quote_mask2);
    printf("literal mask: \n%08x\n%08x\n", literal_mask1, literal_mask2);
    __mmask32 structural_mask1 = 0, whitespace_mask1 = 0;
    __mmask32 structural_mask2 = 0, whitespace_mask2 = 0;
    MercuryJson::extract_structural_whitespace_characters(input1, literal_mask1, structural_mask1, whitespace_mask1);
    MercuryJson::extract_structural_whitespace_characters(input2, literal_mask2, structural_mask2, whitespace_mask2);
    printf("structural mask: \n%08x\n%08x\n", structural_mask1, structural_mask2);
    printf("whitespace mask: \n%08x\n%08x\n", whitespace_mask1, whitespace_mask2);
    prev_mask = 0;
    __mmask32 pseudo_mask1 = MercuryJson::extract_pseudo_structural_mask(structural_mask1, whitespace_mask1, quote_mask1, literal_mask1, prev_mask);
    __mmask32 pseudo_mask2 = MercuryJson::extract_pseudo_structural_mask(structural_mask2, whitespace_mask2, quote_mask2, literal_mask2, prev_mask);
    printf("pseudo mask: \n%08x\n%08x\n", pseudo_mask1, pseudo_mask2);
    size_t indices[64], base = 0;
    MercuryJson::construct_structural_character_pointers(pseudo_mask1, 0, indices, base);
    MercuryJson::construct_structural_character_pointers(pseudo_mask2, 32, indices, base);
    for (size_t i = 0; i < base; ++i) printf("%lu[%c] ", indices[i], buf[indices[i]]);
    printf("\n");
}

void test_extract_warp_mask() {
    size_t size;
    const char * buf = read_file("data/test_extract_escape_mask.json", size);
    printf("%lu: %s\n", size, buf);
    __m256i _input1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(buf));
    __m256i _input2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(buf+32));
    MercuryJson::Warp input1(_input2, _input1);
    __m256i _input3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(buf+64));
    __m256i _input4 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(buf+128));
    MercuryJson::Warp input2(_input4, _input3);
    u_int64_t prev_mask = 0;
    u_int64_t escape_mask1 = MercuryJson::extract_escape_mask(input1, prev_mask);
    u_int64_t escape_mask2 = MercuryJson::extract_escape_mask(input2, prev_mask);
    printf("escape mask: \n%016llx\n%016llx\n", static_cast<unsigned long long>(escape_mask1), static_cast<unsigned long long>(escape_mask2));
    prev_mask = 0;
    u_int64_t quote_mask1 = 0, quote_mask2 = 0;
    u_int64_t literal_mask1 = MercuryJson::extract_literal_mask(input1, escape_mask1, prev_mask, quote_mask1);
    u_int64_t literal_mask2 = MercuryJson::extract_literal_mask(input2, escape_mask2, prev_mask, quote_mask2);
    printf("quote mask: \n%016llx\n%016llx\n", static_cast<unsigned long long>(quote_mask1), static_cast<unsigned long long>(quote_mask2));
    printf("literal mask: \n%016llx\n%016llx\n", static_cast<unsigned long long>(literal_mask1), static_cast<unsigned long long>(literal_mask2));
    u_int64_t structural_mask1 = 0, whitespace_mask1 = 0;
    u_int64_t structural_mask2 = 0, whitespace_mask2 = 0;
    MercuryJson::extract_structural_whitespace_characters(input1, literal_mask1, structural_mask1, whitespace_mask1);
    MercuryJson::extract_structural_whitespace_characters(input2, literal_mask2, structural_mask2, whitespace_mask2);
    printf("structural mask: \n%016llx\n%016llx\n", static_cast<unsigned long long>(structural_mask1), static_cast<unsigned long long>(structural_mask2));
    printf("whitespace mask: \n%016llx\n%016llx\n", static_cast<unsigned long long>(whitespace_mask1), static_cast<unsigned long long>(whitespace_mask2));
    prev_mask = 0;
    u_int64_t pseudo_mask1 = MercuryJson::extract_pseudo_structural_mask(structural_mask1, whitespace_mask1, quote_mask1, literal_mask1, prev_mask);
    u_int64_t pseudo_mask2 = MercuryJson::extract_pseudo_structural_mask(structural_mask2, whitespace_mask2, quote_mask2, literal_mask2, prev_mask);
    printf("pseudo mask: \n%016llx\n%016llx\n", static_cast<unsigned long long>(pseudo_mask1), static_cast<unsigned long long>(pseudo_mask2));
    size_t indices[128], base = 0;
    MercuryJson::construct_structural_character_pointers(pseudo_mask1, 0, indices, base);
    MercuryJson::construct_structural_character_pointers(pseudo_mask2, 64, indices, base);
    for (size_t i = 0; i < base; ++i) printf("%lu[%c] ", indices[i], buf[indices[i]]);
    printf("\n");
}