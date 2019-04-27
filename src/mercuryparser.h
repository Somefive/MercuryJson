#ifndef MERCURYJSON_MERCURYPARSER_H
#define MERCURYJSON_MERCURYPARSER_H

#include <immintrin.h>

#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "block_allocator.hpp"
#include "flags.h"


namespace MercuryJson {
    /* Stage 1 */
    struct Warp {
        __m256i lo, hi;

        Warp(const __m256i &h, const __m256i &l) : hi(h), lo(l) {}

        explicit Warp(const char *address) {
            lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(address));
            hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(address + 32));
        }
    };

    u_int64_t extract_escape_mask(const Warp &raw, u_int64_t *prev_odd_backslash_ending_mask);

    u_int64_t extract_literal_mask(
            const Warp &raw, u_int64_t escape_mask, u_int64_t *prev_literal_ending, u_int64_t *quote_mask);
    void extract_structural_whitespace_characters(
            const Warp &raw, u_int64_t literal_mask, u_int64_t *structural_mask, u_int64_t *whitespace_mask);
    u_int64_t extract_pseudo_structural_mask(
            u_int64_t structural_mask, u_int64_t whitespace_mask, u_int64_t quote_mask, u_int64_t literal_mask,
            u_int64_t *prev_pseudo_structural_end_mask);
    void construct_structural_character_pointers(
            u_int64_t pseudo_structural_mask, size_t offset, size_t *indices, size_t *base);

    __mmask32 extract_escape_mask(__m256i raw, __mmask32 *prev_odd_backslash_ending_mask);
    __mmask32 extract_literal_mask(
            __m256i raw, __mmask32 escape_mask, __mmask32 *prev_literal_ending, __mmask32 *quote_mask);
    void extract_structural_whitespace_characters(
            __m256i raw, __mmask32 literal_mask, __mmask32 *structural_mask, __mmask32 *whitespace_mask);
    __mmask32 extract_pseudo_structural_mask(
            __mmask32 structural_mask, __mmask32 whitespace_mask, __mmask32 quote_mask, __mmask32 literal_mask,
            __mmask32 *prev_pseudo_structural_end_mask);
    void construct_structural_character_pointers(
            __mmask32 pseudo_structural_mask, size_t offset, size_t *indices, size_t *base);

    /* Stage 2 */
    struct JsonValue;

//    typedef std::map<std::string_view, JsonValue> JsonObject;
//    typedef std::vector<std::pair<std::string_view, JsonValue *>> JsonObject;
//    typedef std::vector<JsonValue *> JsonArray;
    struct JsonObject {
        char *key;
        JsonValue *value;
        JsonObject *next;

        explicit JsonObject(char *key, JsonValue *value, JsonObject *next = nullptr)
                : key(key), value(value), next(next) {}
    };

    struct JsonArray {
        JsonValue *value;
        JsonArray *next;

        explicit JsonArray(JsonValue *value, JsonArray *next = nullptr)
                : value(value), next(next) {}
    };

    struct JsonValue {
        enum { TYPE_NULL, TYPE_BOOL, TYPE_STR, TYPE_OBJ, TYPE_ARR, TYPE_INT, TYPE_DEC } type;
        union {
            bool boolean;
            const char *str;
            JsonObject *object;
            JsonArray *array;
            long long int integer;
            double decimal;
        };

        explicit JsonValue() : type(TYPE_NULL) {}
        explicit JsonValue(bool value) : type(TYPE_BOOL), boolean(value) {}
        explicit JsonValue(const char *value) : type(TYPE_STR), str(value) {}
        explicit JsonValue(JsonObject *value) : type(TYPE_OBJ), object(value) {}
        explicit JsonValue(JsonArray *value) : type(TYPE_ARR), array(value) {}
        explicit JsonValue(long long int value) : type(TYPE_INT), integer(value) {}
        explicit JsonValue(double value) : type(TYPE_DEC), decimal(value) {}
    };

    class JSON {
    public:
        char *input;
        size_t input_len, num_indices;
        size_t *indices;
        const size_t *idx_ptr;
#if ALLOC_PARSED_STR
        char *literals;
#endif

        [[noreturn]] void _error(const char *expected, char encountered, size_t index);

        JsonValue *_parse_value();
        JsonValue *_parse_object();
        JsonValue *_parse_array();

        BlockAllocator<JsonValue> allocator;

        char *_parse_str(size_t idx);

    public:
        JsonValue *document;

        JSON(char *document, size_t size, bool manual_construct = false);

        void exec_stage1();
        void exec_stage2();

        ~JSON();
    };

    bool parse_true(const char *s, size_t offset = 0U);
    bool parse_false(const char *s, size_t offset = 0U);
    void parse_null(const char *s, size_t offset = 0U);
    std::variant<double, long long int> parse_number(const char *s, bool *is_decimal, size_t offset = 0U);

    void parse_str_per_bit(const char *src, char *dest, size_t *len = nullptr, size_t offset = 0U);
    void parse_str_naive(char *src, char *dest = nullptr, size_t *len = nullptr, size_t offset = 0U);
    void parse_str_avx(char *src, char *dest = nullptr, size_t *len = nullptr, size_t offset = 0U);
    __m256i translate_escape_characters(__m256i input);
    void deescape(Warp &input, u_int64_t escaper_mask);

    inline __m256i convert_to_mask(u_int32_t input);

    void __printChar_m256i(__m256i raw);
    void __printChar(Warp &raw);

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
}

#endif // MERCURYJSON_MERCURYPARSER_H
