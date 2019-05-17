#include "mercuryparser.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <sstream>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "constants.h"
#include "parsestring.h"


namespace MercuryJson {

    void __printChar(Warp &raw) {
        auto *vals = reinterpret_cast<uint8_t *>(&raw);
        for (size_t i = 0; i < 64; ++i) printf("%2x(%c) ", vals[i], vals[i]);
        printf("\n");
    }

    static const uint64_t kEvenMask64 = 0x5555555555555555U;
    static const uint64_t kOddMask64 = ~kEvenMask64;

    // @formatter:off
    uint64_t extract_escape_mask(const Warp &raw, uint64_t *prev_odd_backslash_ending_mask) {
        uint64_t backslash_mask = __cmpeq_mask(raw, '\\');
        uint64_t start_backslash_mask = backslash_mask & ~(backslash_mask << 1U);

        uint64_t even_start_backslash_mask = (start_backslash_mask & kEvenMask64) ^ *prev_odd_backslash_ending_mask;
        uint64_t even_carrier_backslash_mask = even_start_backslash_mask + backslash_mask;
        uint64_t even_escape_mask = (even_carrier_backslash_mask ^ backslash_mask) & kOddMask64;

        uint64_t odd_start_backslash_mask = (start_backslash_mask & kOddMask64) ^ *prev_odd_backslash_ending_mask;
        uint64_t odd_carrier_backslash_mask = odd_start_backslash_mask + backslash_mask;
        uint64_t odd_escape_mask = (odd_carrier_backslash_mask ^ backslash_mask) & kEvenMask64;

        uint64_t odd_backslash_ending_mask = odd_carrier_backslash_mask < odd_start_backslash_mask;
        *prev_odd_backslash_ending_mask = odd_backslash_ending_mask;

        return even_escape_mask | odd_escape_mask;
    }
    // @formatter:on

    uint64_t extract_literal_mask(
            const Warp &raw, uint64_t escape_mask, uint64_t *prev_literal_ending, uint64_t *quote_mask) {
        uint64_t _quote_mask = __cmpeq_mask(raw, '"') & ~escape_mask;
        uint64_t literal_mask = _mm_cvtsi128_si64(
                _mm_clmulepi64_si128(_mm_set_epi64x(0ULL, _quote_mask), _mm_set1_epi8(0xFF), 0));
        literal_mask ^= *prev_literal_ending;
        *quote_mask = _quote_mask;
        *prev_literal_ending = static_cast<uint64_t>(static_cast<int64_t>(literal_mask) >> 63);
        return literal_mask;
    }

    void extract_structural_whitespace_characters(
            const Warp &raw, uint64_t literal_mask, uint64_t *structural_mask, uint64_t *whitespace_mask) {
        const __m256i upper_lookup = _mm256_setr_epi8(8, 0, 17, 2, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0,
                                                      8, 0, 17, 2, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0);
        const __m256i lower_lookup = _mm256_setr_epi8(16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 4, 1, 12, 0, 0,
                                                      16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 4, 1, 12, 0, 0);

        __m256i hi_upper_index = _mm256_shuffle_epi8(upper_lookup, _mm256_and_si256(_mm256_srli_epi16(raw.hi, 4),
                                                                                    _mm256_set1_epi8(0x7F)));
        __m256i hi_lower_index = _mm256_shuffle_epi8(lower_lookup, raw.hi);
        __m256i hi_character_label = _mm256_and_si256(hi_upper_index, hi_lower_index);

        __m256i lo_upper_index = _mm256_shuffle_epi8(upper_lookup, _mm256_and_si256(_mm256_srli_epi16(raw.lo, 4),
                                                                                    _mm256_set1_epi8(0x7F)));
        __m256i lo_lower_index = _mm256_shuffle_epi8(lower_lookup, raw.lo);
        __m256i lo_character_label = _mm256_and_si256(lo_upper_index, lo_lower_index);

        __m256i hi_whitespace_mask = _mm256_and_si256(hi_character_label, _mm256_set1_epi8(0x18));
        __m256i lo_whitespace_mask = _mm256_and_si256(lo_character_label, _mm256_set1_epi8(0x18));
        *whitespace_mask = ~(__cmpeq_mask(hi_whitespace_mask, lo_whitespace_mask, 0) | literal_mask);

        __m256i hi_structural_mask = _mm256_and_si256(hi_character_label, _mm256_set1_epi8(0x7));
        __m256i lo_structural_mask = _mm256_and_si256(lo_character_label, _mm256_set1_epi8(0x7));
        *structural_mask = ~(__cmpeq_mask(hi_structural_mask, lo_structural_mask, 0) | literal_mask);
    }

    uint64_t extract_pseudo_structural_mask(
            uint64_t structural_mask, uint64_t whitespace_mask, uint64_t quote_mask, uint64_t literal_mask,
            uint64_t *prev_pseudo_structural_end_mask) {
        uint64_t st_ws = structural_mask | whitespace_mask;
        structural_mask |= quote_mask;
        uint64_t pseudo_structural_mask = (st_ws << 1U) | *prev_pseudo_structural_end_mask;
        *prev_pseudo_structural_end_mask = st_ws >> 63U;
        pseudo_structural_mask &= (~whitespace_mask) & (~literal_mask);
        structural_mask |= pseudo_structural_mask;
        structural_mask &= ~(quote_mask & ~literal_mask);
        return structural_mask;
    }

    const size_t kStructuralUnrollCount = 8;

    inline void construct_structural_character_pointers(
            uint64_t pseudo_structural_mask, size_t offset, size_t *indices, size_t *base) {
        size_t next_base = *base + __builtin_popcountll(pseudo_structural_mask);
        while (pseudo_structural_mask) {
            for (size_t i = 0; i < kStructuralUnrollCount; ++i) {
                indices[*base + i] = offset + _tzcnt_u64(pseudo_structural_mask);
                pseudo_structural_mask = _blsr_u64(pseudo_structural_mask);
            }
            *base += kStructuralUnrollCount;
        }
        *base = next_base;
    }

    void __printChar_m256i(__m256i raw) {
        auto *vals = reinterpret_cast<uint8_t *>(&raw);
        for (size_t i = 0; i < 32; ++i) printf("%2x(%c) ", vals[i], vals[i]);
        printf("\n");
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

    void __error_maybe_escape(char *context, size_t *length, char ch) {
        if (ch == '\0') {
            context[(*length)++] = '"';
        } else if (ch == '\t' || ch == '\n' || ch == '\b') {
            context[(*length)++] = '\\';
            switch (ch) {
                case '\t':
                    context[(*length)++] = 't';
                    break;
                case '\n':
                    context[(*length)++] = 'n';
                    break;
                case '\b':
                    context[(*length)++] = 'b';
                    break;
                default:
                    break;
            }
        } else {
            context[(*length)++] = ch;
        }
    }

    [[noreturn]] void __error(const std::string &message, const char *input, size_t offset) {
        static const size_t context_len = 20;
        char *context = new char[(2 * context_len + 1) * 4];  // add space for escaped chars
        size_t length = 0;
        if (offset > context_len) {
            context[0] = context[1] = context[2] = '.';
            length = 3;
        }
        for (size_t i = offset > context_len ? offset - context_len : 0U; i < offset; ++i)
            __error_maybe_escape(context, &length, input[i]);
        size_t left = length;
        bool end = false;
        for (size_t i = offset; i < offset + context_len; ++i) {
            if (input[i] == '\0') {
                end = true;
                break;
            }
            __error_maybe_escape(context, &length, input[i]);
        }
        if (!end) {
            context[length] = context[length + 1] = context[length + 2] = '.';
            length += 3;
        }
        context[length] = 0;
        std::stringstream stream;
        stream << message << std::endl;
        stream << "context: " << context << std::endl;
        delete[] context;
        stream << "         " << std::string(left, ' ') << "^";
        throw std::runtime_error(stream.str());
    }

    long long int parse_number(const char *input, bool *is_decimal, size_t offset) {
#if NO_PARSE_NUMBER
        *is_decimal = false;
        return 0LL;
#endif
        const char *s = input + offset;
        long long int integer = 0LL;
        double decimal = 0.0;
        bool negative = false, _is_decimal = false;
        if (*s == '-') {
            ++s;
            negative = true;
        }
        if (*s == '0') {
            ++s;
            if (*s >= '0' && *s <= '9')
                __error("numbers cannot have leading zeros", input, offset);
        } else {
            if (*s < '0' || *s > '9')
                MercuryJson::__error("numbers must have integer parts", input, offset);
#if PARSE_NUMBER_AVX
            while (_all_digits(s)) {
                integer = integer * 100000000 + _parse_eight_digits(s);
                s += 8;
            }
#endif
            while (*s >= '0' && *s <= '9')
                integer = integer * 10 + (*s++ - '0');
        }
        if (s - input - offset > 18)
            __error("integer part too large", input, offset);
        if (*s == '.') {
            _is_decimal = true;
            decimal = integer;
            double multiplier = 0.1;
            ++s;
#if PARSE_NUMBER_AVX
            while (_all_digits(s)) {
                decimal += _parse_eight_digits(s) * multiplier * 0.0000001;  // 7 digits
                multiplier *= 0.00000001;  // 8 digits
                s += 8;
            }
#endif
            while (*s >= '0' && *s <= '9') {
                decimal += (*s++ - '0') * multiplier;
                multiplier *= 0.1;
            }if (multiplier == 0.1)
                __error("excessive characters at end of number", input, s - input - 1);
        }
        if (*s == 'e' || *s == 'E') {
            if (!_is_decimal) {
                _is_decimal = true;
                decimal = integer;
            }
            ++s;
            bool negative_exp = false;
            if (*s == '-') {
                negative_exp = true;
                ++s;
            } else if (*s == '+') ++s;
            double exponent = 0.0;
            if (*s < '0' || *s > '9')
                MercuryJson::__error("numbers must not have null exponents", input, s - input);
            do {
                exponent = exponent * 10.0 + (*s++ - '0');
            } while (*s >= '0' && *s <= '9');
            if (negative_exp) exponent = -exponent;
            if (exponent < -308 || exponent > 308)
                __error("decimal exponent out of range", input, offset);
            decimal *= pow(10.0, exponent);
        }
        if (!kStructuralOrWhitespace[*s])
            __error("excessive characters at end of number", input, s - input);
        *is_decimal = _is_decimal;
        if (negative) {
            if (_is_decimal) return plain_convert(-decimal);
            else return -integer;
        } else {
            if (_is_decimal) return plain_convert(decimal);
            else return integer;
        }
    }

    bool parse_true(const char *s, size_t offset) {
        uint32_t literal;
        memcpy(&literal, s + offset, 4);
        uint32_t target = 0x65757274;
        if (target != literal || !kStructuralOrWhitespace[s[offset + 4]])
            __error("invalid true value", s, offset);
        return true;
    }

    bool parse_false(const char *s, size_t offset) {
        uint64_t literal;
        memcpy(&literal, s + offset, 5);
        uint64_t target = 0x00000065736c6166;
        uint64_t mask = 0x000000ffffffffff;
        if (target != (literal & mask) || !kStructuralOrWhitespace[s[offset + 5]])
            __error("invalid false value", s, offset);
        return false;
    }

    void parse_null(const char *s, size_t offset) {
        uint32_t literal;
        memcpy(&literal, s + offset, 4);
        uint32_t target = 0x6c6c756e;
        if (target != literal || !kStructuralOrWhitespace[s[offset + 4]])
            __error("invalid null value", s, offset);
    }

    [[noreturn]] void JSON::_error(const char *expected, char encountered, size_t index) {
        std::stringstream stream;
        char _encounter[3];
        size_t len = 0;
        __error_maybe_escape(_encounter, &len, encountered);
        _encounter[len] = 0;
        stream << "expected " << expected << " at index " << index << ", but encountered '" << _encounter << "'";
        MercuryJson::__error(stream.str(), input, index);
    }

#if PARSE_STR_NUM_THREADS

    void JSON::_thread_parse_str(size_t pid) {
//        auto start_time = std::chrono::steady_clock::now();
        size_t idx;
        const size_t *idx_ptr = indices + pid * num_indices / PARSE_STR_NUM_THREADS;  // deliberate shadowing
        const size_t *end_ptr = indices + (pid + 1) * num_indices / PARSE_STR_NUM_THREADS;
        char ch;
        do {
            idx = *idx_ptr++;
            ch = input[idx];
            if (ch == '"') {
# if ALLOC_PARSED_STR
                char *dest = literals + idx + 1;
# else
                char *dest = input + idx + 1;
# endif

# if PARSE_STR_MODE == 2
                parse_str_per_bit(input, dest, nullptr, idx + 1);
# elif PARSE_STR_MODE == 1
                parse_str_avx(input, dest, nullptr, idx + 1);
# elif PARSE_STR_MODE == 0
                parse_str_naive(input, dest, nullptr, idx + 1);
# endif
            }
        } while (idx_ptr != end_ptr);
//        std::chrono::duration<double> runtime = std::chrono::steady_clock::now() - start_time;
//        printf("parse str thread %lu: %.6lf\n", pid, runtime.count());
    }

#endif

    JSON::JSON(char *document, size_t size, bool manual_construct) : allocator(size) {
        input = document;
        input_len = size;
        this->document = nullptr;

        idx_ptr = indices = aligned_malloc<size_t>(size);
        num_indices = 0;
#if ALLOC_PARSED_STR
        literals = static_cast<char *>(aligned_malloc(size));
#endif

        if (!manual_construct) {
            exec_stage1();
            exec_stage2();
        }
    }

    void JSON::exec_stage1() {
        uint64_t prev_escape_mask = 0;
        uint64_t prev_quote_mask = 0;
        uint64_t prev_pseudo_mask = 1;  // initial value set to 1 to allow literals at beginning of input
        uint64_t quote_mask, structural_mask, whitespace_mask;
        uint64_t pseudo_mask = 0;
        size_t offset = 0;
        for (; offset < input_len; offset += 64) {
            Warp warp(input + offset);
            uint64_t escape_mask = extract_escape_mask(warp, &prev_escape_mask);
            uint64_t literal_mask = extract_literal_mask(warp, escape_mask, &prev_quote_mask, &quote_mask);

            // Dump pointers for *previous* iteration.
            construct_structural_character_pointers(pseudo_mask, offset - 64, indices, &num_indices);

            extract_structural_whitespace_characters(warp, literal_mask, &structural_mask, &whitespace_mask);
            pseudo_mask = extract_pseudo_structural_mask(
                    structural_mask, whitespace_mask, quote_mask, literal_mask, &prev_pseudo_mask);
        }
        // Dump pointers for the final iteration.
        construct_structural_character_pointers(pseudo_mask, offset - 64, indices, &num_indices);
        if (num_indices == 0 || input[indices[num_indices - 1]] != '\0')  // Ensure '\0' is added to indices.
            indices[num_indices++] = offset - 64 + strlen(input + offset - 64);
        if (prev_quote_mask != 0)
            throw std::runtime_error("unclosed quotation marks");
    }

    void JSON::exec_stage2() {
//        std::chrono::time_point<std::chrono::steady_clock> start_time;
//        std::chrono::duration<double> runtime;
#if PARSE_STR_NUM_THREADS
//        start_time = std::chrono::steady_clock::now();
        std::thread parse_str_threads[PARSE_STR_NUM_THREADS];
        for (size_t i = 0; i < PARSE_STR_NUM_THREADS; ++i)
            parse_str_threads[i] = std::thread(&JSON::_thread_parse_str, this, i);
//        runtime = std::chrono::steady_clock::now() - start_time;
//        printf("thread spawn: %.6lf\n", runtime.count());
#endif

//        start_time = std::chrono::steady_clock::now();
#if SHIFT_REDUCE_PARSER
        document = _shift_reduce_parsing();  // final index is '\0'
#else
        document = _parse_value();
#endif
        char ch = input[*idx_ptr];
        if (ch != '\0') _error("file end", ch, *idx_ptr);
//        runtime = std::chrono::steady_clock::now() - start_time;
//        printf("parse document: %.6lf\n", runtime.count());
#if PARSE_STR_NUM_THREADS
//        start_time = std::chrono::steady_clock::now();
        for (std::thread &thread : parse_str_threads)
            thread.join();
//        runtime = std::chrono::steady_clock::now() - start_time;
//        printf("wait join: %.6lf\n", runtime.count());
#endif
        aligned_free(indices);
        indices = nullptr;
    }

    JSON::~JSON() {
        if (indices != nullptr) aligned_free(indices);
#if ALLOC_PARSED_STR
        aligned_free(literals);
#endif
    }
}
