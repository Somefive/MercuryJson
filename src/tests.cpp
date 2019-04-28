#include "tests.h"

#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <string>
#include <vector>

#include "mercuryparser.h"
#include "tape.h"
#include "utils.h"


template <typename T>
inline std::string to_binary(T x) {
    auto s = std::bitset<sizeof(T) * 8>(x).to_string();
    std::reverse(s.begin(), s.end());
    return s;
}

template <typename T>
inline void print(const char *name, std::vector<T> masks) {
    printf("%10s mask: ", name);
    for (auto &mask : masks)
        printf("%s", to_binary(mask).c_str());
    printf("\n");
}

void test_extract_mask() {
    size_t size;
    char *buf = read_file("data/test_extract_escape_mask.json", &size);
    if (buf[size - 1] == '\n') buf[--size] = 0;
    printf("%15lu: %s\n", size, buf);

    std::vector<__mmask32> escape_masks;
    std::vector<__mmask32> quote_masks;
    std::vector<__mmask32> literal_masks;
    std::vector<__mmask32> structural_masks;
    std::vector<__mmask32> whitespace_masks;
    std::vector<__mmask32> pseudo_masks;
    size_t indices[64], base = 0;

    __mmask32 prev_escape_mask = 0;
    __mmask32 prev_quote_mask = 0;
    __mmask32 prev_pseudo_mask = 0;
    for (size_t offset = 0; offset < size; offset += 32) {
        __m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(buf + offset));
        __mmask32 escape_mask = MercuryJson::extract_escape_mask(input, &prev_escape_mask);
        __mmask32 quote_mask = 0;
        __mmask32 literal_mask = MercuryJson::extract_literal_mask(input, escape_mask, &prev_quote_mask, &quote_mask);
        __mmask32 structural_mask = 0, whitespace_mask = 0;
        MercuryJson::extract_structural_whitespace_characters(input, literal_mask, &structural_mask, &whitespace_mask);
        __mmask32 pseudo_mask = MercuryJson::extract_pseudo_structural_mask(
                structural_mask, whitespace_mask, quote_mask, literal_mask, &prev_pseudo_mask);
        MercuryJson::construct_structural_character_pointers(pseudo_mask, offset, indices, &base);

        escape_masks.push_back(escape_mask);
        quote_masks.push_back(quote_mask);
        literal_masks.push_back(literal_mask);
        structural_masks.push_back(structural_mask);
        whitespace_masks.push_back(whitespace_mask);
        pseudo_masks.push_back(pseudo_mask);
    }

    printf("%15c  ", ' ');
    for (int i = 0; i < escape_masks.size(); ++i)
        printf("[%30c]", ' ');
    printf("\n");
    print("escape", escape_masks);
    print("quote", quote_masks);
    print("literal", literal_masks);
    print("structural", structural_masks);
    print("whitespace", whitespace_masks);
    print("pseudo", pseudo_masks);
    for (size_t i = 0; i < base; ++i) printf("%lu[%c] ", indices[i], buf[indices[i]]);
    printf("\n");
}

void test_extract_warp_mask() {
    size_t size;
    char *buf = read_file("data/test_extract_escape_mask.json", &size);
    if (buf[size - 1] == '\n') buf[--size] = 0;
    printf("%15lu: %s\n", size, buf);

    std::vector<u_int64_t> escape_masks;
    std::vector<u_int64_t> quote_masks;
    std::vector<u_int64_t> literal_masks;
    std::vector<u_int64_t> structural_masks;
    std::vector<u_int64_t> whitespace_masks;
    std::vector<u_int64_t> pseudo_masks;
    size_t indices[64], base = 0;

    u_int64_t prev_escape_mask = 0;
    u_int64_t prev_quote_mask = 0;
    u_int64_t prev_pseudo_mask = 0;
    for (size_t offset = 0; offset < size; offset += 64) {
        __m256i _input1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(buf + offset));
        __m256i _input2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(buf + offset + 32));
        MercuryJson::Warp input(_input2, _input1);
        u_int64_t escape_mask = MercuryJson::extract_escape_mask(input, &prev_escape_mask);
        u_int64_t quote_mask = 0;
        u_int64_t literal_mask = MercuryJson::extract_literal_mask(input, escape_mask, &prev_quote_mask, &quote_mask);
        u_int64_t structural_mask = 0, whitespace_mask = 0;
        MercuryJson::extract_structural_whitespace_characters(input, literal_mask, &structural_mask, &whitespace_mask);
        u_int64_t pseudo_mask = MercuryJson::extract_pseudo_structural_mask(
                structural_mask, whitespace_mask, quote_mask, literal_mask, &prev_pseudo_mask);
        MercuryJson::construct_structural_character_pointers(pseudo_mask, offset, indices, &base);

        escape_masks.push_back(escape_mask);
        quote_masks.push_back(quote_mask);
        literal_masks.push_back(literal_mask);
        structural_masks.push_back(structural_mask);
        whitespace_masks.push_back(whitespace_mask);
        pseudo_masks.push_back(pseudo_mask);
    }

    printf("%15c  ", ' ');
    for (int i = 0; i < escape_masks.size(); ++i)
        printf("[%62c]", ' ');
    printf("\n");
    print("escape", escape_masks);
    print("quote", quote_masks);
    print("literal", literal_masks);
    print("structural", structural_masks);
    print("whitespace", whitespace_masks);
    print("pseudo", pseudo_masks);
    for (size_t i = 0; i < base; ++i) printf("%lu[%c] ", indices[i], buf[indices[i]]);
    printf("\n");
}

void test_tfn_value() {
    bool ok = true;
    auto good_literals = {"null      ", "null\0", "true,      ", "true}      ", "false]      ", "false\n"};
    for (auto &str : good_literals) {
        try {
            switch (str[0]) {
                case 'n':
                    MercuryJson::parse_null(str);
                    break;
                case 't':
                    MercuryJson::parse_true(str);
                    break;
                case 'f':
                    MercuryJson::parse_false(str);
                    break;
                default:
                    break;
            }
        } catch (const std::runtime_error &err) {
            printf("test_tfn_value: error parsing good literal \"%s\", : %s\n", str, err.what());
            ok = false;
        }
    }
    auto bad_literals = {"nulll", "true\"", "fals ", "trueu", "nah", "null\\"};
    for (auto &str : bad_literals) {
        try {
            switch (str[0]) {
                case 'n':
                    MercuryJson::parse_null(str);
                    break;
                case 't':
                    MercuryJson::parse_true(str);
                    break;
                case 'f':
                    MercuryJson::parse_false(str);
                    break;
                default:
                    break;
            }
            printf("test_tfn_value: no error parsing bad literal \"%s\"\n", str);
            ok = false;
        } catch (const std::runtime_error &err) {}
    }
    if (ok) printf("test_tfn_value: passed\n");
}

using namespace MercuryJson;

void test_parse(bool print) {
    size_t size;
//    char *buf = read_file("data/test_extract_escape_mask.json", &size);
//    char *buf = read_file("data/pp.json", &size);
    char *buf = read_file("data/non-ascii.json", &size);
    auto json = JSON(buf, size);
    if (print) print_json(json.document);
}

void test_parse_str_naive() {
    char text[256] = R"("something\tto parse\nnextLine here with lots of escape\\\\;")";
    std::cout << "Original:" << std::endl << text << std::endl;
    parse_str_naive(text + 1);
    char *p = text + 1;
    std::cout << "Parsed: " << std::endl << p << std::endl;
    std::cout << std::endl;
}

void test_parse_str_avx() {
    char text[256] = R"("something\tto parse\nnextLine here with lots of escape\\\\;\n)"
                     R"(this is a cross boundary test!\\!!!!", this should be invisible\tOh!)";
    // char text[512] = "\"fLA[/wsxV\\r&Io#`G\\t5XBZM|;/|HvoxPWE\\n0Rf%K:\\tOcaRD)DWag/0aJ<\\\\o3Lia!,P2^84(O)T4g'UpK*O0:\\\\\\raxOR\"!";
    std::cout << "Original:" << std::endl << text << std::endl;
    parse_str_avx(text + 1);
    char *p = text + 1;
    std::cout << "Parsed: " << std::endl << p << std::endl;
    std::cout << std::endl;
}

void test_parse_str_per_bit() {
    char text[256] = R"("something\tto parse\nnextLine here with lots of escape\\\\;\n)"
                     R"(this is a cross boundary test!\\!!!!", this should be invisible\tOh!)";
    std::cout << "Original:" << std::endl << text << std::endl;
    char dest[256];
    parse_str_per_bit(text + 1, dest);
    std::cout << "Parsed: " << std::endl << dest << std::endl;
    std::cout << std::endl;
}

char *generate_random_string(size_t length) {
    char *text = reinterpret_cast<char *>(aligned_malloc(ALIGNMENT_SIZE, length));
    char *base = text;
    *base++ = '"';
    for (size_t i = 0; i < length - 4; ++i) {
        char c = rand() % (127 - 32) + 32;
        switch (c) {
            case 'b':
                // case 'f':
            case 'n':
            case 'r':
            case 't':
            case '"':
            case '\\':
                *base++ = '\\';
                ++i;
                break;
            default:
                break;
        }
        *base++ = c;
    }
    *base++ = '"';
    *base++ = '!';
    *base++ = '\0';
    return text;
}

void test_parse_string() {
    srand(time(nullptr));
    clock_t t_baseline = 0, t_avx = 0, t_per_bit = 0;
    for (int i = 0; i < 10; ++i) {
        auto length = static_cast<size_t>(1e8);
        char *text_naive = generate_random_string(length);
        char *text_avx = reinterpret_cast<char *>(aligned_malloc(ALIGNMENT_SIZE, length));
        char *p_per_bit = reinterpret_cast<char *>(aligned_malloc(ALIGNMENT_SIZE, length));
        strcpy(text_avx, text_naive);
        printf("test[%d]: ", i);
        fflush(stdout);
        size_t l_naive, l_avx, l_per_bit;
        clock_t t0 = clock();
        parse_str_naive(text_naive + 1, nullptr, &l_naive);
        char *p_naive = text_naive + 1;
        clock_t t1 = clock();
        parse_str_per_bit(text_avx + 1, p_per_bit, &l_per_bit);
        clock_t t2 = clock();
        parse_str_avx(text_avx + 1, nullptr, &l_avx);
        char *p_avx = text_avx + 1;
        clock_t t3 = clock();
        t_baseline += (t1 - t0);
        t_avx += (t2 - t1);
        t_per_bit += (t3 - t2);
        int result_avx = strncmp(p_naive, p_avx, l_naive);
        int result_per_bit = strncmp(p_naive, p_per_bit, l_naive);
        if (result_avx == 0 && result_per_bit == 0) printf("passed\n");
        else {
            if (result_avx != 0) printf("parse_str_avx incorrect: %d\n", result_avx);
            if (result_per_bit != 0) printf("parse_str_per_bit incorrect: %d\n", result_per_bit);
        }
        aligned_free(text_naive);
        aligned_free(text_avx);
        aligned_free(p_per_bit);
    }
    printf("baseline: %.4f sec, avx: %.4f sec, per_bit: %.4f sec\n",
           static_cast<float>(t_baseline) / CLOCKS_PER_SEC,
           static_cast<float>(t_avx) / CLOCKS_PER_SEC,
           static_cast<float>(t_per_bit) / CLOCKS_PER_SEC);
}

#define xstr(x) ___str___(x)
#define ___str___(x) #x

void test_parse_float() {
#define FLOAT_VAL 0.00001234556
    double expected = FLOAT_VAL;
    const char *s = xstr(FLOAT_VAL);
    bool is_decimal;
    auto ret = parse_number(s, &is_decimal);
    if (!is_decimal) printf("test_parse_float: is_decimal flag incorrect\n");
    else {
        auto val = std::get<double>(ret);
        if (fabs(val - 0.00001234556) > 1e-10) {
            printf("test_parse_float: expected %.10lf, received %.10lf\n", expected, val);
        } else printf("test_parse_float: passed\n");
    }
#undef FLOAT_VAL
}

void test_translate() {
    const char *s = R"(/0"1\2b3f4n5r6t7t8r9nAfBbC\D"E/F)";
    __m256i input = Warp(s).lo;
    printf("origin: ");
    __printChar_m256i(input);
    printf("new:    ");
    __printChar_m256i(translate_escape_characters(input));
}

void test_remove_escaper() {
    const char *s = R"(\tabc\t\n\\\t\t\n\\\t\t\n\\\t\t\n\\\t\t\n\\\t\t\n\\t\t\n\\\t\t\n\\t\t\n\\\t\t\n\\)";
    printf("s: %s\n", s);
    Warp input(s);
    printf("input: ");
    __printChar(input);
    u_int64_t prev_odd_backslash_ending_mask = 0ULL;
    u_int64_t escape_mask = extract_escape_mask(input, &prev_odd_backslash_ending_mask);
    u_int64_t escaper_mask = (escape_mask >> 1U) | (prev_odd_backslash_ending_mask << 63U);
    auto masks = std::vector<u_int64_t>();
    masks.push_back(escaper_mask);
    print("escaper", masks);
    deescape(input, escaper_mask);
    printf("output: ");
    __printChar(input);
}

void test_tape(const char *filename) {
    size_t size;
    char *input = read_file(filename, &size);
    auto json = MercuryJson::JSON(input, size, true);
    json.exec_stage1();
    Tape tape(size, size);
    TapeWriter tape_writer(&tape, const_cast<char *>(json.input), json.indices);
    tape_writer._parse_value();
    tape.print_json();
}
