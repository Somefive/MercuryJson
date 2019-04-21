#ifndef MERCURYJSON_MERCURYPARSER_H
#define MERCURYJSON_MERCURYPARSER_H

#include <deque>
#include <map>

using std::size_t;

namespace MercuryJson {
    /* Stage 1 */
    struct Warp {
        __m256i lo, hi;

        Warp(const __m256i &h, const __m256i &l) : hi(h), lo(l) {}
        Warp(const char * address) {
            lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(address));
            hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(address+32));
        }
    };

    template <bool include_backslash = false>
    u_int64_t extract_escape_mask(const Warp &raw, u_int64_t *prev_odd_backslash_ending_mask);

    // explicit instantiation declaration
    extern template u_int64_t extract_escape_mask<true>(const Warp &raw, u_int64_t *prev_odd_backslash_ending_mask);
    extern template u_int64_t extract_escape_mask<false>(const Warp &raw, u_int64_t *prev_odd_backslash_ending_mask);

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
    typedef std::map<std::string, JsonValue> JsonObject;
    typedef std::deque<JsonValue> JsonArray;

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

    char *parseStr(char *s);
    bool parseTrue(const char *s);
    bool parseFalse(const char *s);
    void parseNull(const char *s);

    void parse(char *input, size_t len, size_t *indices);

    JsonValue parseJson(char *document, size_t size);

    char *parseStrAVX(char *s);
    __m256i translate_escape_characters(__m256i input);
    void deescape(Warp &input, u_int64_t escaper_mask);
    
    inline __m256i convert_to_mask(u_int32_t input);

    void __print_m256i(__m256i raw);
    void __printChar_m256i(__m256i raw);
    void __print(Warp &raw);
    void __printChar(Warp &raw);
}

#endif // MERCURYJSON_MERCURYPARSER_H
