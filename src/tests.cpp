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

#include "tests.h"
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
        MercuryJson::parse_null("null      ");
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

void test_parse(bool print) {
    size_t size;
//    char *buf = read_file("data/test_extract_escape_mask.json", &size);
//    char *buf = read_file("data/pp.json", &size);
    char *buf = read_file("data/demographic_statistics_by_zipcode.json", &size);
    auto json = JSON(buf, size);
    if (print) print_json(json.document);
}

void test_parseStr() {
    char text[256] = R"("something\tto parse\nnextLine here with lots of escape\\\\;")";
    std::cout << "Original:" << std::endl << text << std::endl;
    char *p = parse_str_naive(text);
    std::cout << "Parsed: " << std::endl << p << std::endl;
    std::cout << std::endl;
}

void test_parseStrAVX() {
    char text[256] = R"("something\tto parse\nnextLine here with lots of escape\\\\;\n)"
                     R"(this is a cross boundary test!\\!!!!", this should be invisible\tOh!)";
    // char text[512] = "\"fLA[/wsxV\\r&Io#`G\\t5XBZM|;/|HvoxPWE\\n0Rf%K:\\tOcaRD)DWag/0aJ<\\\\o3Lia!,P2^84(O)T4g'UpK*O0:\\\\\\raxOR\"!";
    std::cout << "Original:" << std::endl << text << std::endl;
    char *p = parse_str_avx(text);
    std::cout << "Parsed: " << std::endl << p << std::endl;
    std::cout << std::endl;
}

char *generate_randomString(size_t length) {
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

void test_parseString() {
    srand(time(nullptr));
    clock_t t_baseline = 0, t_avx = 0;
    for (int i = 0; i < 10; ++i) {
        size_t length = static_cast<size_t>(1e8);
        // const char *base = generate_randomString(length);
        // printf("base: %s\n", base);
        // char *text = reinterpret_cast<char *>(aligned_malloc(ALIGNMENT_SIZE, length));
        char *text = generate_randomString(length);
        char *text2 = reinterpret_cast<char *>(aligned_malloc(ALIGNMENT_SIZE, length));
        // strcpy(text, base);
        // printf("base: %s\n", text);
        strcpy(text2, text);
        printf("test[%d]: ", i);
        fflush(stdout);
        clock_t t0 = clock();
        char *p1 = parse_str_naive(text);
        clock_t t1 = clock();
        char *p2 = parse_str_avx(text2);
        clock_t t2 = clock();
        t_baseline += (t1 - t0);
        t_avx += (t2 - t1);
        // printf("%s\n\n\n%s\n\n\n", p1, p2);
        int result = strcmp(p1, p2);
        if (result == 0) printf("passed\n");
        else printf("incorrect: %d\n", result);
        aligned_free(text);
        aligned_free(text2);
    }
    printf("baseline: %.4f sec, avx: %.4f sec\n", float(t_baseline) / CLOCKS_PER_SEC, float(t_avx) / CLOCKS_PER_SEC);
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
