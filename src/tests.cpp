#include <cstdio>
#include <immintrin.h>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstring>

#include "mercuryparser.h"
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
    try {
        MercuryJson::parseNull("null      ");
    } catch (const std::runtime_error &err) {
        printf("error: %s\n", err.what());
    }
}

using namespace MercuryJson;

void print_indent(int indent) {
    if (indent > 0) {
        std::cout << std::string(indent, ' ');
    }
}

void print_json(const JsonValue &value, int indent = 0) {
    int cnt;
    switch (value.type) {
        case JsonValue::TYPE_NULL:
            std::cout << "null";
            break;
        case JsonValue::TYPE_BOOL:
            if (value.boolean) std::cout << "true";
            else std::cout << "false";
            break;
        case JsonValue::TYPE_STR:
            std::cout << "\"" << value.str << "\"";
            break;
        case JsonValue::TYPE_OBJ:
            std::cout << "{" << std::endl;
            cnt = 0;
            for (const auto &it : *value.object) {
                print_indent(indent + 2);
                std::cout << "\"" << it.first << "\": ";
                print_json(it.second, indent + 2);
                if (cnt + 1 < value.object->size())
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
            for (const auto &it : *value.array) {
                print_indent(indent + 2);
                print_json(it, indent + 2);
                if (cnt + 1 < value.object->size())
                    std::cout << ",";
                std::cout << std::endl;
                ++cnt;
            }
            print_indent(indent);
            std::cout << "]";
            break;
        case JsonValue::TYPE_INT:
            std::cout << value.integer;
            break;
        case JsonValue::TYPE_DEC:
            std::cout << value.decimal;
            break;
    }
}

void test_parse() {
    size_t size;
//    char *buf = read_file("data/test_extract_escape_mask.json", &size);
//    char *buf = read_file("data/pp.json", &size);
    char *buf = read_file("data/demographic_statistics_by_zipcode.json", &size);
    auto json = parseJson(buf, size);
    // print_json(json);
}

void test_parseStr() {
    char text[256] = "\"something\\tto parse\\nnextLine here with lots of escape\\\\\\\\;\"";
    char *p = parseStr(text);
    std::cout << p << std::endl;
}

void test_parseStrAVX() {
    char text[256] = "\"something\\tto parse\\nnextLine here with lots of escape\\\\\\\\;\\nthis is a cross boundary test!\\\\!!!!\", this should be invisible\\tOh!";
    char *p = parseStrAVX(text+1);
    std::cout << p << std::endl;
}

char *generate_randomString(size_t length) {
    char *text = reinterpret_cast<char *>(aligned_malloc(ALIGNMENT_SIZE, length));
    char *base = text;
    *base++ = '"';
    for (size_t i=0; i<length-4; ++i) {
        char c = rand() % (127 - 32) + 32;
        switch (c)
        {
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

void test_parseString() {
    srand(time(0));
    clock_t t_baseline = 0, t_avx = 0;
    for (int i = 0; i < 10; ++i) {
        size_t length = 1e8;
        // const char *base = generate_randomString(length);
        // printf("base: %s\n", base);
        // char *text = reinterpret_cast<char *>(aligned_malloc(ALIGNMENT_SIZE, length));
        char *text = generate_randomString(length);
        char *text2 = reinterpret_cast<char *>(aligned_malloc(ALIGNMENT_SIZE, length));
        // strcpy(text, base);
        strcpy(text2, text);
        printf("test begin\n");
        clock_t t0 = clock();
        char *p1 = parseStr(text);
        clock_t t1 = clock();
        char *p2 = parseStrAVX(text2);
        clock_t t2 = clock();
        t_baseline += (t1-t0);
        t_avx += (t2-t1);
        // printf("%s\n\n\n%s\n\n\n", p1, p2);
        printf("test[%d] consistent: %d\n", i, strcmp(p1, p2));
        aligned_free(text);
        aligned_free(text2);
    }
    printf("baseline: %.4f sec, avx: %.4f sec\n", float(t_baseline) / CLOCKS_PER_SEC, float(t_avx) / CLOCKS_PER_SEC);
}