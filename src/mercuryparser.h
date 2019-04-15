#ifndef MERCURYJSON_MERCURYPARSER_H
#define MERCURYJSON_MERCURYPARSER_H

#include <deque>
#include <map>

namespace MercuryJson {
    /* Stage 1 */
    struct Warp {
        __m256i hi, lo;

        Warp(const __m256i &h, const __m256i &l) : hi(h), lo(l) {}
    };

    u_int64_t extract_escape_mask(const Warp &raw, u_int64_t &prev_odd_backslash_ending_mask);
    u_int64_t extract_literal_mask(
            const Warp &raw, u_int64_t escape_mask, u_int64_t &prev_literal_ending, u_int64_t &quote_mask);
    void extract_structural_whitespace_characters(
            const Warp &raw, u_int64_t literal_mask, u_int64_t &structural_mask, u_int64_t &whitespace_mask);
    u_int64_t extract_pseudo_structural_mask(
            u_int64_t structural_mask, u_int64_t whitespace_mask, u_int64_t quote_mask, u_int64_t literal_mask,
            u_int64_t &prev_pseudo_structural_end_mask);
    void construct_structural_character_pointers(
            u_int64_t pseudo_structural_mask, size_t offset, size_t *indices, size_t &base);

    __mmask32 extract_escape_mask(__m256i raw, __mmask32 &prev_odd_backslash_ending_mask);
    __mmask32 extract_literal_mask(
            __m256i raw, __mmask32 escape_mask, __mmask32 &prev_literal_ending, __mmask32 &quote_mask);
    void extract_structural_whitespace_characters(
            __m256i raw, __mmask32 literal_mask, __mmask32 &structural_mask, __mmask32 &whitespace_mask);
    __mmask32 extract_pseudo_structural_mask(
            __mmask32 structural_mask, __mmask32 whitespace_mask, __mmask32 quote_mask, __mmask32 literal_mask,
            __mmask32 &prev_pseudo_structural_end_mask);
    void construct_structural_character_pointers(
            __mmask32 pseudo_structural_mask, size_t offset, size_t *indices, size_t &base);

    /* Stage 2 */
    struct JsonValue;
    typedef std::map<std::string, JsonValue> JsonObject;
    typedef std::deque<JsonValue> JsonArray;

    union Numerical {
        long long int integer;
        double decimal;
    };

    struct JsonValue {
        enum { TYPE_NULL, TYPE_BOOL, TYPE_STR, TYPE_OBJ, TYPE_ARR, TYPE_INT, TYPE_DEC, TYPE_CHAR } type;
        union {
            bool boolean;
            char *str;
            JsonObject *object;
            JsonArray *array;
            long long int integer;
            double decimal;

            char structural;
        };

        static JsonValue create() { return JsonValue({.type=JsonValue::TYPE_NULL}); }

        static JsonValue create(bool value) { return JsonValue({.type=JsonValue::TYPE_BOOL, {.boolean=value}}); }

        static JsonValue create(char *value) { return JsonValue({.type=JsonValue::TYPE_STR, {.str=value}}); }

        static JsonValue create(JsonObject *value) { return JsonValue({.type=JsonValue::TYPE_OBJ, {.object=value}}); }

        static JsonValue create(JsonArray *value) { return JsonValue({.type=JsonValue::TYPE_ARR, {.array=value}}); }

        static JsonValue create(long long int value) {
            return JsonValue({.type=JsonValue::TYPE_INT, {.integer=value}});
        }

        static JsonValue create(double value) { return JsonValue({.type=JsonValue::TYPE_DEC, {.decimal=value}}); }

        static JsonValue create(char c) { return JsonValue{.type=TYPE_CHAR, {.structural=c}}; }
    };

    char *parseStr(const char *s, char *&buffer);
    bool parseTrue(const char *s);
    bool parseFalse(const char *s);
    void parseNull(const char *s);

    void parse(char *input, size_t len, size_t *indices);
}

#endif // MERCURYJSON_MERCURYPARSER_H
