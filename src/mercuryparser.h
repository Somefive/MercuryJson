#ifndef MERCURYJSON_MERCURYPARSER_H
#define MERCURYJSON_MERCURYPARSER_H

#include <deque>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <variant>

#include "utils.h"

#define USE_BLOCK_ALLOCATOR 1

using std::size_t;

namespace MercuryJson {
    /* Stage 1 */
    struct Warp {
        __m256i lo, hi;

        Warp(const __m256i &h, const __m256i &l) : hi(h), lo(l) {}

        Warp(const char *address) {
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

    template <typename default_class>
    class BlockedAllocator {
    private:
#if USE_BLOCK_ALLOCATOR
        char *mem, *ptr;
        std::vector<char *> all_memory;
        size_t block_size, allocated;
        static constexpr size_t alignment = sizeof(default_class);
#endif

    public:
        BlockedAllocator(size_t block_size) {
#if USE_BLOCK_ALLOCATOR
            block_size = round_up(block_size, alignment);
            ptr = mem = reinterpret_cast<char *>(aligned_malloc(alignment, block_size));
            allocated = 0;
            this->block_size = block_size;
#endif
        }

        ~BlockedAllocator() {
#if USE_BLOCK_ALLOCATOR
            // if (all_memory.size() > 0)
            //     printf("%lu blocks allocated\n", all_memory.size() + 1);
            aligned_free(mem);
            for (void *p : all_memory)
                aligned_free(p);
#endif
        }

        template <typename T = default_class, typename ...Args>
        T *construct(Args ...args) {
#if USE_BLOCK_ALLOCATOR
            if (allocated + sizeof(T) > block_size) {
                all_memory.push_back(mem);
                allocated = 0;
                ptr = mem = reinterpret_cast<char *>(aligned_malloc(alignment, block_size));
            }
            static constexpr size_t size = round_up(sizeof(T), alignment);
            T *ret = new(ptr) T(std::forward<Args>(args)...);
            ptr += size;
            allocated += size;
#else
            T *ret = new T(std::forward<Args>(args)...);
#endif
            return ret;
        }
    };

    class JSON {
    private:
        char *input;
        size_t input_len;
        size_t *indices, *idx_ptr;

        void __error(const char *expected, char encountered, size_t index);

        JsonValue *_parseValue();
        JsonValue *_parseObject();
        JsonValue *_parseArray();

        BlockedAllocator<JsonValue> allocator;

    public:
        JsonValue *document;

        JSON(char *document, size_t size, bool manual_construct = false);

        void exec_stage1();
        void exec_stage2();

        ~JSON();
    };

    char *parseStr(char *s, size_t *len = nullptr);
    bool parseTrue(const char *s);
    bool parseFalse(const char *s);
    void parseNull(const char *s);

    char *parseStrAVX(char *s, size_t *len = nullptr);
    __m256i translate_escape_characters(__m256i input);
    void deescape(Warp &input, u_int64_t escaper_mask);

    inline __m256i convert_to_mask(u_int32_t input);

    void __print_m256i(__m256i raw);
    void __printChar_m256i(__m256i raw);
    void __print(Warp &raw);
    void __printChar(Warp &raw);

}

#endif // MERCURYJSON_MERCURYPARSER_H
