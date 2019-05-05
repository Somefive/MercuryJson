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


namespace MercuryJson {

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
        if (ch == '\t' || ch == '\n' || ch == '\b' || ch == '\0') {
            context[(*length)++] = '\\';
//            context[(*length)++] = '[';
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
                case '\0':
                    context[(*length)++] = '0';
                    break;
                default:
                    break;
            }
//            context[(*length)++] = ']';
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

    std::variant<double, long long int> parse_number(const char *input, bool *is_decimal, size_t offset) {
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
#if PARSE_NUMBER_AVX
            while (_all_digits(s)) {
                integer = integer * 100000000 + _parse_eight_digits(s);
                s += 8;
            }
#endif
            while (*s >= '0' && *s <= '9')
                integer = integer * 10 + (*s++ - '0');
        }
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
            }
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
            while (*s >= '0' && *s <= '9')
                exponent = exponent * 10.0 + (*s++ - '0');
            if (negative_exp) exponent = -exponent;
            decimal *= pow(10.0, exponent);
        }
        *is_decimal = _is_decimal;
        if (negative) {
            if (_is_decimal) return -decimal;
            else return -integer;
        } else {
            if (_is_decimal) return decimal;
            else return integer;
        }
    }

    void parse_str_naive(const char *src, char *dest, size_t *len, size_t offset) {
        bool escape = false;
        char *ptr = dest == nullptr ? const_cast<char *>(src) : dest, *base = ptr;
        for (const char *end = src + offset; escape || *end != '"'; ++end) {
            if (escape) {
                switch (*end) {
                    case '"':
                        *ptr++ = '"';
                        break;
                    case '\\':
                        *ptr++ = '\\';
                        break;
                    case '/':
                        *ptr++ = '/';
                        break;
                    case 'b':
                        *ptr++ = '\b';
                        break;
                    case 'f':
                        *ptr++ = '\f';
                        break;
                    case 'n':
                        *ptr++ = '\n';
                        break;
                    case 'r':
                        *ptr++ = '\r';
                        break;
                    case 't':
                        *ptr++ = '\t';
                        break;
                    case 'u':
                        // TODO: Should deal with unicode encoding
                        *ptr++ = '\\';
                        *ptr++ = 'u';
                        break;
                    default:
                        __error("invalid escape sequence", src, end - src);
                }
                escape = false;
            } else {
                if (*end == '\\') escape = true;
                else *ptr++ = *end;
            }
        }
        *ptr++ = 0;
        if (len != nullptr) *len = ptr - base;
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
        uint64_t prev_pseudo_mask = 0;
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

    void parse_str_per_bit(const char *src, char *dest, size_t *len, size_t offset) {
        const char *_src = src;
        src += offset;
        char *base = dest;
        while (true) {
            __m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest), input);
            __mmask32 backslash_mask = __cmpeq_mask(input, '\\');
            __mmask32 quote_mask = __cmpeq_mask(input, '"');

            if (((backslash_mask - 1) & quote_mask) != 0) {
                // quotes first
                size_t quote_offset = _tzcnt_u32(quote_mask);
                dest[quote_offset] = 0;
                if (len != nullptr) *len = (dest - base) + quote_offset;
                break;
            } else if (((quote_mask - 1) & backslash_mask) != 0) {
                // backslash first
                size_t backslash_offset = _tzcnt_u32(backslash_mask);
                uint8_t escape_char = src[backslash_offset + 1];
                if (escape_char == 'u') {
                    // TODO: deal with unicode characters
                    memcpy(dest + backslash_offset, src + backslash_offset, 6);
                    src += backslash_offset + 6;
                    dest += backslash_offset + 6;
                } else {
                    uint8_t escaped = kEscapeMap[escape_char];
                    if (escaped == 0U)
                        __error("invalid escape character '" + std::string(1, escape_char) + "'", _src, offset);
                    dest[backslash_offset] = kEscapeMap[escape_char];
                    src += backslash_offset + 2;
                    dest += backslash_offset + 1;
                }
            } else {
                // nothing here
                src += sizeof(__mmask32);
                dest += sizeof(__mmask32);
            }
        }
    }

    void parse_str_avx(const char *src, char *dest, size_t *len, size_t offset) {
#if PARSE_STR_32BIT
        typedef __mmask32 mask_t;
# define tzcnt _tzcnt_u32
# define blsr _blsr_u32
#else
        typedef uint64_t mask_t;
# define tzcnt _tzcnt_u64
# define blsr _blsr_u64
#endif

        const char *_src = src;
        src += offset;
        if (dest == nullptr) dest = const_cast<char *>(src);
        char *base = dest;
        mask_t prev_odd_backslash_ending_mask = 0ULL;
        while (true) {
#if PARSE_STR_32BIT
            __m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
#else
            Warp input(src);
#endif
            mask_t escape_mask = extract_escape_mask(input, &prev_odd_backslash_ending_mask);
            mask_t quote_mask = __cmpeq_mask(input, '"') & (~escape_mask);

            size_t ending_offset = tzcnt(quote_mask);

            if (ending_offset < sizeof(mask_t) * 8) {
                size_t last_offset = 0, length;
                while (true) {
                    size_t this_offset = tzcnt(escape_mask);
                    if (this_offset >= ending_offset) {
                        memmove(dest, src + last_offset, ending_offset - last_offset);
                        dest[ending_offset - last_offset] = '\0';
                        if (len != nullptr) *len = (dest - base) + (ending_offset - last_offset);
                        break;
                    }
                    length = this_offset - last_offset;
                    char escaper = src[this_offset];
                    memmove(dest, src + last_offset, length);
                    dest += length;
                    *(dest - 1) = kEscapeMap[escaper];
                    last_offset = this_offset + 1;
                    escape_mask = blsr(escape_mask);
                }
                break;
            } else {
#if PARSE_STR_FULLY_AVX
                /* fully-AVX version */
                __m256i lo_mask = convert_to_mask(escape_mask);
                __m256i hi_mask = convert_to_mask(escape_mask >> 32U);
                // mask ? translated : original
                __m256i lo_trans = translate_escape_characters(input.lo);
                __m256i hi_trans = translate_escape_characters(input.hi);
                input.lo = _mm256_blendv_epi8(lo_trans, input.lo, lo_mask);
                input.hi = _mm256_blendv_epi8(hi_trans, input.hi, hi_mask);
                uint64_t escaper_mask = (escape_mask >> 1U) | (prev_odd_backslash_ending_mask << 63U);

                deescape(input, escaper_mask);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest), input.lo);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest + 32), input.hi);
                dest += 64 - _mm_popcnt_u64(escaper_mask);
                src += 64;
#else
                size_t last_offset = 0, length;
                while (true) {
                    size_t this_offset = tzcnt(escape_mask);
                    length = this_offset - last_offset;
                    char escaper = src[this_offset];
                    memmove(dest, src + last_offset, length);
                    dest += length;
                    if (this_offset >= ending_offset) break;
                    *(dest - 1) = kEscapeMap[escaper];
                    last_offset = this_offset + 1;
                    escape_mask = blsr(escape_mask);
                }
                src += sizeof(mask_t) * 8;
#endif
            }
        }
#undef tzcnt
#undef blsr
    }

    inline __m256i convert_to_mask(uint32_t input) {
        /* Create a mask based on each bit of `input`.
         *                    input : [0-31]
         * _mm256_set1_epi32(input) : [0-7] [8-15] [16-23] [24-31] * 8
         *      _mm256_shuffle_epi8 : [0-7] * 8 [8-15] * 8 ...
         *         _mm256_and_si256 : [0] [1] [2] [3] [4] ...
         *        _mm256_cmpeq_epi8 : 0xFF for 1-bits, 0x00 for 0-bits
         */
        const __m256i projector = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                                   1, 1, 1, 1, 1, 1, 1, 1,
                                                   2, 2, 2, 2, 2, 2, 2, 2,
                                                   3, 3, 3, 3, 3, 3, 3, 3);
        const __m256i masker = _mm256_setr_epi8(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
        const __m256i zeros = _mm256_set1_epi8(0);
        return _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(
                _mm256_set1_epi32(input), projector), masker), zeros);
    }

    inline uint64_t __extract_highestbit_pext(const Warp &input, int shift, uint64_t escaper_mask) {
        uint64_t lo = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(input.lo, shift)));
        uint64_t hi = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(input.hi, shift)));
        return _pext_u64(((hi << 32U) | lo), escaper_mask);
    }

    inline __m256i __expand(uint32_t input) {
        /* 0x00 for 1-bits, 0x01 for 0-bits */
        const __m256i ones = _mm256_set1_epi8(1);
        return _mm256_add_epi8(convert_to_mask(input), ones);
    }

    inline __m256i __reconstruct(uint32_t h0, uint32_t h1, uint32_t h2, uint32_t h3,
                                 uint32_t h4, uint32_t h5, uint32_t h6, uint32_t h7) {
        /* Reconstruct a __m256i with 8-bit numbers specified by bits from h0~h7 */
        __m256i result = __expand(h0);
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h1), 1));
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h2), 2));
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h3), 3));
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h4), 4));
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h5), 5));
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h6), 6));
        result = _mm256_or_si256(result, _mm256_slli_epi16(__expand(h7), 7));
        return result;
    }

    void deescape(Warp &input, uint64_t escaper_mask) {
        /* Remove 8-bit characters specified by `escaper_mask` */
        uint64_t nonescaper_mask = ~escaper_mask;
        // Obtain each bit from each 8-bit character, keeping only non-escapers
        uint64_t h7 = __extract_highestbit_pext(input, 0, nonescaper_mask);
        uint64_t h6 = __extract_highestbit_pext(input, 1, nonescaper_mask);
        uint64_t h5 = __extract_highestbit_pext(input, 2, nonescaper_mask);
        uint64_t h4 = __extract_highestbit_pext(input, 3, nonescaper_mask);
        uint64_t h3 = __extract_highestbit_pext(input, 4, nonescaper_mask);
        uint64_t h2 = __extract_highestbit_pext(input, 5, nonescaper_mask);
        uint64_t h1 = __extract_highestbit_pext(input, 6, nonescaper_mask);
        uint64_t h0 = __extract_highestbit_pext(input, 7, nonescaper_mask);
        // _mm256_packus_epi32
        input.lo = __reconstruct(h0, h1, h2, h3, h4, h5, h6, h7);
        input.hi = __reconstruct(
                h0 >> 32U, h1 >> 32U, h2 >> 32U, h3 >> 32U, h4 >> 32U, h5 >> 32U, h6 >> 32U, h7 >> 32U);
    }

    // 1. '/'   0x2f 0x2f
    // 2. '""'  0x22 0x22
    // 4. '\'   0x5c 0x5c
    // 8. 'b'   0x62 0x08
    // 16.'f'   0x66 0x0c
    // 32.'n'   0x6e 0x0a
    // 64.'r'   0x72 0x0d
    //128.'t'   0x74 0x09
    // TODO: Unicode literals and non-escapable character validation
    __m256i translate_escape_characters(__m256i input) {
        const __m256i hi_lookup = _mm256_setr_epi8(0, 0, 0x03, 0, 0, 0x04, 0x38, 0xc0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0x03, 0, 0, 0x04, 0x38, 0xc0, 0, 0, 0, 0, 0, 0, 0, 0);
        const __m256i lo_lookup = _mm256_setr_epi8(0, 0, 0x4a, 0, 0x80, 0, 0x10, 0, 0, 0, 0, 0, 0x04, 0, 0x20, 0x01,
                                                   0, 0, 0x4a, 0, 0x80, 0, 0x10, 0, 0, 0, 0, 0, 0x04, 0, 0x20, 0x01);
        __m256i hi_index = _mm256_shuffle_epi8(hi_lookup,
                                               _mm256_and_si256(_mm256_srli_epi16(input, 4), _mm256_set1_epi8(0x7F)));
        __m256i lo_index = _mm256_shuffle_epi8(lo_lookup, input);
        __m256i character_class = _mm256_and_si256(hi_index, lo_index);
        const __m256i trans_1 = _mm256_setr_epi8(0, 0x2f, 0x22, 0, 0x5c, 0, 0, 0, 0x08, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0x2f, 0x22, 0, 0x5c, 0, 0, 0, 0x08, 0, 0, 0, 0, 0, 0, 0);
        const __m256i trans_2 = _mm256_setr_epi8(0, 0x0c, 0x0a, 0, 0x0d, 0, 0, 0, 0x09, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0x0c, 0x0a, 0, 0x0d, 0, 0, 0, 0x09, 0, 0, 0, 0, 0, 0, 0);
        __m256i trans = _mm256_or_si256(
                _mm256_shuffle_epi8(trans_1, character_class),
                _mm256_shuffle_epi8(trans_2, _mm256_and_si256(
                        _mm256_srli_epi16(character_class, 4), _mm256_set1_epi8(0x7F))));
        return trans;
    }
}
