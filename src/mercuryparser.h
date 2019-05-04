#ifndef MERCURYJSON_MERCURYPARSER_H
#define MERCURYJSON_MERCURYPARSER_H

#include <immintrin.h>
#include <string.h>

#include <string>
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

    uint64_t extract_escape_mask(const Warp &raw, uint64_t *prev_odd_backslash_ending_mask);

    uint64_t extract_literal_mask(
            const Warp &raw, uint64_t escape_mask, uint64_t *prev_literal_ending, uint64_t *quote_mask);
    void extract_structural_whitespace_characters(
            const Warp &raw, uint64_t literal_mask, uint64_t *structural_mask, uint64_t *whitespace_mask);
    uint64_t extract_pseudo_structural_mask(
            uint64_t structural_mask, uint64_t whitespace_mask, uint64_t quote_mask, uint64_t literal_mask,
            uint64_t *prev_pseudo_structural_end_mask);
    void construct_structural_character_pointers(
            uint64_t pseudo_structural_mask, size_t offset, size_t *indices, size_t *base);

    /* Stage 2 */

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

    struct JsonValue;

    struct JsonObject {
        const char *key;
        JsonValue *value;
        JsonObject *next;

        explicit JsonObject(const char *key, JsonValue *value, JsonObject *next = nullptr)
                : key(key), value(value), next(next) {}
    };

    struct JsonArray {
        JsonValue *value;
        JsonArray *next;

        explicit JsonArray(JsonValue *value, JsonArray *next = nullptr)
                : value(value), next(next) {}
    };

    struct JsonValue {
        enum ValueType : int { TYPE_NULL, TYPE_BOOL, TYPE_STR, TYPE_OBJ, TYPE_ARR, TYPE_INT, TYPE_DEC } type;
        union {
            bool boolean;
            const char *str;
            JsonObject *object;
            JsonArray *array;
            long long int integer;
            double decimal;
        };

        //@formatter:off
        explicit JsonValue() : type(TYPE_NULL) {}
        explicit JsonValue(bool value) : type(TYPE_BOOL), boolean(value) {}
        explicit JsonValue(const char *value) : type(TYPE_STR), str(value) {}
        explicit JsonValue(JsonObject *value) : type(TYPE_OBJ), object(value) {}
        explicit JsonValue(JsonArray *value) : type(TYPE_ARR), array(value) {}
        explicit JsonValue(long long int value) : type(TYPE_INT), integer(value) {}
        explicit JsonValue(double value) : type(TYPE_DEC), decimal(value) {}
        //@formatter:on
    };

    void __error_maybe_escape(char *context, size_t *length, char ch);
    [[noreturn]] void __error(const std::string &message, const char *input, size_t offset);

    namespace shift_reduce_impl { struct ParseStack; }

    class JSON {
    public:
#if ALLOC_PARSED_STR
        const
#endif
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

#if PARSE_STR_NUM_THREADS
        void _thread_parse_str(size_t pid);
#endif

        void _thread_shift_reduce_parsing(const size_t *idx_begin, const size_t *idx_end,
                                          shift_reduce_impl::ParseStack *stack);

        JsonValue *_shift_reduce_parsing();

    public:
        JsonValue *document;

        JSON(char *document, size_t size, bool manual_construct = false);

        void exec_stage1();
        void exec_stage2();

        ~JSON();
    };

    inline char *JSON::_parse_str(size_t idx) {
#if ALLOC_PARSED_STR
        char *dest = literals + idx + 1;
#else
        char *dest = input + idx + 1;
#endif

#if !PARSE_STR_NUM_THREADS
        # if PARSE_STR_MODE == 2
        parse_str_per_bit(input, dest, nullptr, idx + 1);
# elif PARSE_STR_MODE == 1
        parse_str_avx(input, dest, nullptr, idx + 1);
# elif PARSE_STR_MODE == 0
        parse_str_naive(input, dest, nullptr, idx + 1);
# endif
#endif
        return dest;
    }

    void print_json(MercuryJson::JsonValue *value, int indent = 0);

    bool parse_true(const char *s, size_t offset = 0U);
    bool parse_false(const char *s, size_t offset = 0U);
    void parse_null(const char *s, size_t offset = 0U);
    std::variant<double, long long int> parse_number(const char *s, bool *is_decimal, size_t offset = 0U);

    void parse_str_per_bit(const char *src, char *dest, size_t *len = nullptr, size_t offset = 0U);
    void parse_str_naive(const char *src, char *dest = nullptr, size_t *len = nullptr, size_t offset = 0U);
    void parse_str_avx(const char *src, char *dest = nullptr, size_t *len = nullptr, size_t offset = 0U);
    __m256i translate_escape_characters(__m256i input);
    void deescape(Warp &input, uint64_t escaper_mask);

    inline __m256i convert_to_mask(uint32_t input);

    void __printChar_m256i(__m256i raw);
    void __printChar(Warp &raw);

    inline uint64_t __cmpeq_mask(const __m256i raw_hi, const __m256i raw_lo, char c) {
        const __m256i vec_c = _mm256_set1_epi8(c);
        uint64_t hi = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(raw_hi, vec_c)));
        uint64_t lo = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(raw_lo, vec_c)));
        return (hi << 32U) | lo;
    }

    inline uint64_t __cmpeq_mask(const Warp &raw, char c) {
        return __cmpeq_mask(raw.hi, raw.lo, c);
    }

    inline __mmask32 __cmpeq_mask(__m256i raw, char c) {
        __m256i vec_c = _mm256_set1_epi8(c);
        return static_cast<__mmask32>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(raw, vec_c)));
    }

    inline bool _all_digits(const char *s) {
        __m64 val;
        memcpy(&val, s, 8);
        __m64 base = _mm_sub_pi8(val, _mm_set1_pi8('0'));
        __m64 basecmp = _mm_subs_pu8(base, _mm_set1_pi8(9));
        return _mm_cvtm64_si64(basecmp) == 0;
    }

    inline uint32_t _parse_eight_digits(const char *s) {
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
}

#endif // MERCURYJSON_MERCURYPARSER_H
