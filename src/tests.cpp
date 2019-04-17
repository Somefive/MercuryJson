#include <cstdio>
#include <immintrin.h>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <string>
#include <vector>

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
            for (const auto& it : *value.object) {
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
            for (const auto& it : *value.array) {
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
        case JsonValue::TYPE_CHAR:
            std::cout << value.structural;
            break;
    }
}

void test_parse() {
    size_t size;
//    char *buf = read_file("data/test_extract_escape_mask.json", &size);
//    char *buf = read_file("data/pp.json", &size);
    char *buf = read_file("data/demographic_statistics_by_zipcode.json", &size);
    auto json = parseJson(buf, size);
    print_json(json);
}
