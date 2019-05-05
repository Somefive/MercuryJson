#include "mercuryparser.h"

#include <assert.h>

#include <thread>
#include <vector>

#include "block_allocator.hpp"
#include "flags.h"


namespace MercuryJson::shift_reduce_impl {
    struct JsonPartialValue;

    struct JsonPartialObject {
        const char *key;
        JsonPartialValue *value;
        JsonPartialObject *next;

        JsonPartialObject(const char *key, JsonPartialValue *value, JsonPartialObject *next = nullptr)
                : key(key), value(value), next(next) {}
    };

    struct JsonPartialObjectHead : JsonPartialObject {
        JsonPartialObject *final;

        JsonPartialObjectHead(const char *key, JsonPartialValue *value, JsonPartialObject *next = nullptr)
                : JsonPartialObject(key, value, next), final(this) {}
    };

    struct JsonPartialArray {
        JsonPartialValue *value;
        JsonPartialArray *next;

        explicit JsonPartialArray(JsonPartialValue *value, JsonPartialArray *next = nullptr)
                : value(value), next(next) {}
    };

    struct JsonPartialArrayHead : JsonPartialArray {
        JsonPartialArray *final;

        JsonPartialArrayHead(JsonPartialValue *value, JsonPartialArray *next = nullptr)
                : JsonPartialArray(value, next), final(this) {}
    };

    struct JsonPartialValue {
        enum ValueType : int {
            TYPE_NULL, TYPE_BOOL, TYPE_STR, TYPE_OBJ, TYPE_ARR, TYPE_INT, TYPE_DEC,
            TYPE_PARTIAL_OBJ, TYPE_PARTIAL_ARR, TYPE_CHAR
        } type;

        union {
            bool boolean;
            const char *str;
            JsonObject *object;
            JsonArray *array;
            JsonPartialObject *partial_object;
            JsonPartialArray *partial_array;
            long long int integer;
            double decimal;
            char structural;
        };

        //@formatter:off
            explicit JsonPartialValue() : type(TYPE_NULL) {}
            explicit JsonPartialValue(bool value) : type(TYPE_BOOL), boolean(value) {}
            explicit JsonPartialValue(const char *value) : type(TYPE_STR), str(value) {}
            explicit JsonPartialValue(JsonObject *value) : type(TYPE_OBJ), object(value) {}
            explicit JsonPartialValue(JsonArray *value) : type(TYPE_ARR), array(value) {}
            explicit JsonPartialValue(long long int value) : type(TYPE_INT), integer(value) {}
            explicit JsonPartialValue(double value) : type(TYPE_DEC), decimal(value) {}

            explicit JsonPartialValue(JsonPartialObject *value) : type(TYPE_PARTIAL_OBJ), partial_object(value) {}
            explicit JsonPartialValue(JsonPartialArray *value) : type(TYPE_PARTIAL_ARR), partial_array(value) {}
            explicit JsonPartialValue(char value) : type(TYPE_CHAR), structural(value) {}
            //@formatter:on
    };

    void print_json(JsonPartialValue *value, size_t indent = 0) {
        switch (value->type) {
            case JsonPartialValue::TYPE_PARTIAL_OBJ:
            case JsonPartialValue::TYPE_PARTIAL_ARR: {
                print_indent(indent);
                std::cout << "(partial) ";
                JsonValue new_value = *reinterpret_cast<JsonValue *>(value);
                new_value.type = static_cast<JsonValue::ValueType>(static_cast<int>(value->type) - 4);
                print_json(&new_value, indent);
            }
            case JsonPartialValue::TYPE_NULL:
            case JsonPartialValue::TYPE_BOOL:
            case JsonPartialValue::TYPE_STR:
            case JsonPartialValue::TYPE_OBJ:
            case JsonPartialValue::TYPE_ARR:
            case JsonPartialValue::TYPE_INT:
            case JsonPartialValue::TYPE_DEC:
                print_json(reinterpret_cast<JsonValue *>(value), indent);
                break;
            case JsonPartialValue::TYPE_CHAR:
                print_indent(indent);
                std::cout << "'" << value->structural << "'";
                break;
        }
        std::cout << std::endl;
    }

    static const size_t kDefaultStackSize = 1024;

    class ParseStack {
        JsonPartialValue **stack;
        size_t stack_top;
        BlockAllocator<JsonValue> &allocator;

    public:
        ParseStack(BlockAllocator<JsonValue> &allocator, size_t max_size = kDefaultStackSize)
                : allocator(allocator) {
            stack = aligned_malloc<JsonPartialValue *>(max_size);
            stack_top = 0;
        }

        ParseStack(ParseStack &&other) noexcept : allocator(other.allocator) {
            stack = other.stack;
            stack_top = other.stack_top;
            other.stack = nullptr;
        }

        ~ParseStack() {
            aligned_free(stack);
        }

        inline size_t size() const { return stack_top; }

        inline JsonPartialValue *operator [](size_t idx) const { return stack[idx]; }

        template <typename ...Args>
        inline bool _check_stack_top(Args ...) const;

        template <typename ...Args>
        inline bool _check_stack_top(bool, Args ...) const;

        template <typename ...Args>
        inline bool _check_stack_top(char, Args ...) const;

        template <typename ...Args>
        inline bool _check_stack_top(JsonPartialValue::ValueType, Args ...) const;

        template <typename ...Args>
        inline bool check(Args ...args) const {
            if (stack_top < sizeof...(args)) return false;
            return _check_stack_top(args...);
        }

        inline bool check_pos(size_t pos, char ch) const {
            // pos starts from 1, counts from stack top.
            if (stack_top < pos) return false;
            auto *top = stack[stack_top - pos];
            return top->type == JsonPartialValue::TYPE_CHAR && top->structural == ch;
        }

        inline bool check_pos(size_t pos, JsonPartialValue::ValueType type) const {
            // pos starts from 1, counts from stack top.
            if (stack_top < pos) return false;
            auto *top = stack[stack_top - pos];
            return top->type == type;
        }

        inline JsonPartialValue *get(size_t pos) const {
            return stack[stack_top - pos];
        }

        template <typename ...Args>
        inline void push(Args ...args) {
            stack[stack_top++] = allocator.construct<JsonPartialValue>(std::forward<Args>(args)...);
        }

        inline void push(JsonPartialValue *value) {
            stack[stack_top++] = value;
        }

        inline void pop(size_t n) {
            stack_top -= n;
        }

        void print() {
            std::cout << "Stack size: " << stack_top << std::endl;
            for (size_t i = 0; i < stack_top; ++i) {
                std::cout << "Element #" << i << ": ";
                print_json(stack[i]);
            }
        }

        // "{", partial-object, "}"  =>  object
        inline void reduce_object() {
            if (check('{')) {
                // Emtpy object.
                pop(1);
                push(static_cast<JsonObject *>(nullptr));
            } else if (check('{', JsonPartialValue::TYPE_PARTIAL_OBJ)) {
                // Non-empty object.
                auto *obj = reinterpret_cast<JsonObject *>(get(1)->partial_object);
                pop(2);
                push(obj);
            } else {
                // This should not happen when the input is well-formed.
                push('}');
            }
        }

        // "[", partial-array, "]"  =>  array
        inline void reduce_array() {
            if (check('[')) {
                // Emtpy array.
                pop(1);
                push(static_cast<JsonArray *>(nullptr));
            } else if (check('[', true)) {
                // Construct singleton array.
                auto *arr = allocator.construct<JsonArray>(reinterpret_cast<JsonValue *>(get(1)));
                pop(2);
                push(arr);
            } else if (check('[', JsonPartialValue::TYPE_PARTIAL_ARR, ',', true)) {
                // We have to manually match the final element in the array, because we only reduce to partial
                // array on commas (,).
                auto *partial_arr = static_cast<JsonPartialArrayHead *>(get(3)->partial_array);
                partial_arr->final->next = reinterpret_cast<JsonPartialArray *>(
                        allocator.construct<JsonArray>(reinterpret_cast<JsonValue *>(get(1))));
                auto *arr = reinterpret_cast<JsonArray *>(partial_arr);
                pop(4);
                push(arr);
            } else {
                // This should not happen when the input is well-formed.
                push(']');
            }
        }

        // ( [ partial-array ], "," | "[" ), value, ","  =>  partial-array, ","
        inline void reduce_partial_array() {
            if (check('[', true)) {
                // Construct a singleton partial array. Note that previous value must be of complete type,
                // otherwise we might aggressively match partial objects.
                auto *elem = get(1);
                auto *partial_arr = allocator.construct<JsonPartialArrayHead>(elem);
                pop(1);
                push(partial_arr);
                push(',');
            } else if (check(',', true)) {
                // Merge with previous partial array.
                auto *elem = get(1);
                JsonPartialArray *partial_arr;
                if (check_pos(3, JsonPartialValue::TYPE_PARTIAL_ARR)) {
                    partial_arr = allocator.construct<JsonPartialArray>(elem);
                    auto *prev_arr = static_cast<JsonPartialArrayHead *>(get(3)->partial_array);
                    prev_arr->final = prev_arr->final->next = partial_arr;
                    pop(1);  // No need to push ',' --- just re-use the previous one.
                } else {
                    partial_arr = allocator.construct<JsonPartialArrayHead>(elem);
                    pop(1);
                    push(partial_arr);
                    push(',');
                }
            } else {
                push(',');
            }
        }

        // [ partial-object, "," ], string, ":", value  =>  partial-object
        inline bool reduce_partial_object() {
            if (check(JsonPartialValue::TYPE_STR, ':', true)) {
                // Construct singleton partial object.
                auto *key = get(3)->str;
                auto *value = get(1);
                JsonPartialObject *partial_obj;
                pop(3);
                if (check(JsonPartialValue::TYPE_PARTIAL_OBJ, ',')) {
                    // Merge with previous partial object.
                    partial_obj = allocator.construct<JsonPartialObject>(key, value);
                    auto *prev_obj = static_cast<JsonPartialObjectHead *>(get(2)->partial_object);
                    prev_obj->final = prev_obj->final->next = partial_obj;
                    partial_obj = prev_obj;
                    pop(2);
                } else {
                    partial_obj = allocator.construct<JsonPartialObjectHead>(key, value);
                }
                push(partial_obj);
                return true;
            }
            return false;
        }
    };

    template <>
    inline bool ParseStack::_check_stack_top<>() const { return true; }

    // true for any fully-parsed JSON value
    template <typename ...Args>
    inline bool ParseStack::_check_stack_top(bool first, Args ...args) const {
//            assert(first);
        auto *top = stack[stack_top - sizeof...(args) - 1];
        switch (top->type) {
            case JsonPartialValue::TYPE_PARTIAL_OBJ:
            case JsonPartialValue::TYPE_PARTIAL_ARR:
            case JsonPartialValue::TYPE_CHAR:
                return false;
            default:
                break;
        }
        return _check_stack_top(args...);
    }

    template <typename ...Args>
    inline bool ParseStack::_check_stack_top(char first, Args ...args) const {
        auto *top = stack[stack_top - sizeof...(args) - 1];
        if (top->type != JsonPartialValue::TYPE_CHAR || top->structural != first) return false;
        return _check_stack_top(args...);
    }

    template <typename ...Args>
    inline bool ParseStack::_check_stack_top(JsonPartialValue::ValueType first, Args ...args) const {
        auto *top = stack[stack_top - sizeof...(args) - 1];
        if (top->type != first) return false;
        return _check_stack_top(args...);
    }

}

namespace MercuryJson {

#define error(__expected) ({         \
        _error(__expected, ch, idx); \
    })

    void JSON::_thread_shift_reduce_parsing(const size_t *idx_begin, const size_t *idx_end,
                                            shift_reduce_impl::ParseStack *stack) {
        using shift_reduce_impl::JsonPartialValue;
        using shift_reduce_impl::JsonPartialObject;
        using shift_reduce_impl::JsonPartialArray;
        using shift_reduce_impl::ParseStack;

//        auto start_time = std::chrono::steady_clock::now();

        while (idx_begin != idx_end) {
            size_t idx = *idx_begin;
            char ch = input[idx];

            // Shift current value onto stack.
            switch (ch) {
                case '"':
                    stack->push(_parse_str(idx));
                    break;
                case 't':
                    stack->push(parse_true(input, idx));
                    break;
                case 'f':
                    stack->push(parse_false(input, idx));
                    break;
                case 'n':
                    parse_null(input, idx);
                    stack->push();
                    break;
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                case '-': {
                    bool is_decimal;
                    auto ret = parse_number(input, &is_decimal, idx);
                    if (is_decimal) {
                        stack->push(std::get<double>(ret));
                    } else {
                        stack->push(std::get<long long int>(ret));
                    }
                    break;
                }

                case '{':
                case '[':
                case ':':
                    stack->push(ch);
                    break;

                    // Perform reduce on the stack.
                    // There will be at most two consecutive reduce ops:
                    //   1. From a partial array to array, or from a partial object to object.
                    //   2. Merge the previously constructed value with a partial array or partial object.
                case '}':
                    // "{", partial-object, "}"  =>  object
                    stack->reduce_object();
                    break;
                case ']':
                    // "[", partial-array, "]"  =>  array
                    stack->reduce_array();
                    break;
                case ',':
                    // ( [ partial-array ], "," | "[" ), value, ","  =>  partial-array, ","
                    stack->reduce_partial_array();
                    break;
                default:
                    error("JSON value");
            }

            // Possible second step of reduce.
            stack->reduce_partial_object();

            ++idx_begin;
        }

//        std::chrono::duration<double> runtime = std::chrono::steady_clock::now() - start_time;
//        printf("shift-reduce thread: %.6lf, stack size: %lu\n", runtime.count(), stack->size());
    }

    JsonValue *JSON::_shift_reduce_parsing() {
        using shift_reduce_impl::JsonPartialValue;
        using shift_reduce_impl::JsonPartialObjectHead;
        using shift_reduce_impl::JsonPartialArrayHead;
        using shift_reduce_impl::ParseStack;

#if SHIFT_REDUCE_NUM_THREADS > 1
        std::thread shift_reduce_threads[SHIFT_REDUCE_NUM_THREADS - 1];
        std::vector<BlockAllocator<JsonValue>> allocators;
        allocators.reserve(SHIFT_REDUCE_NUM_THREADS - 1);
        std::vector<ParseStack> stacks;
        stacks.reserve(SHIFT_REDUCE_NUM_THREADS);
        stacks.emplace_back(allocator);
        size_t num_indices_per_thread = (num_indices - 1 + SHIFT_REDUCE_NUM_THREADS) / SHIFT_REDUCE_NUM_THREADS;
        for (int i = 0; i < SHIFT_REDUCE_NUM_THREADS - 1; ++i)
            allocators.push_back(allocator.fork(2 * num_indices_per_thread * sizeof(JsonPartialValue)));
        for (int i = 0; i < SHIFT_REDUCE_NUM_THREADS - 1; ++i)
            stacks.emplace_back(allocators[i]);
        for (int i = 0; i < SHIFT_REDUCE_NUM_THREADS - 1; ++i) {
            size_t idx_begin = (num_indices - 1) * (i + 1) / SHIFT_REDUCE_NUM_THREADS;
            size_t idx_end = (num_indices - 1) * (i + 2) / SHIFT_REDUCE_NUM_THREADS;
            shift_reduce_threads[i] = std::thread(&JSON::_thread_shift_reduce_parsing, this,
                                                  indices + idx_begin, indices + idx_end, &stacks[i + 1]);
        }
        size_t idx_end = (num_indices - 1) / SHIFT_REDUCE_NUM_THREADS;
        _thread_shift_reduce_parsing(indices, indices + idx_end, &stacks[0]);

        // Join threads and merge.
        ParseStack &main_stack = stacks[0];
//        main_stack.print();
        for (int i = 0; i < SHIFT_REDUCE_NUM_THREADS - 1; ++i) {
            shift_reduce_threads[i].join();
            ParseStack &merge_stack = stacks[i + 1];
//            merge_stack.print();
            for (size_t idx = 0; idx < merge_stack.size(); ++idx) {
                auto *value = merge_stack[idx];
                switch (value->type) {
                    case JsonPartialValue::TYPE_NULL:
                    case JsonPartialValue::TYPE_BOOL:
                    case JsonPartialValue::TYPE_STR:
                    case JsonPartialValue::TYPE_OBJ:
                    case JsonPartialValue::TYPE_ARR:
                    case JsonPartialValue::TYPE_INT:
                    case JsonPartialValue::TYPE_DEC:
                        main_stack.push(value);
                        break;
                    case JsonPartialValue::TYPE_PARTIAL_OBJ:
                        if (main_stack.check(JsonPartialValue::TYPE_PARTIAL_OBJ, ',')) {
                            // Merge with partial object from previous stack.
                            auto *prev_obj = static_cast<JsonPartialObjectHead *>(main_stack.get(2)->partial_object);
                            prev_obj->final->next = value->partial_object;
                            prev_obj->final = static_cast<JsonPartialObjectHead *>(value->partial_object)->final;
                            main_stack.pop(1);
                        } else {
                            main_stack.push(value);
                        }
                        break;
                    case JsonPartialValue::TYPE_PARTIAL_ARR:
                        if (main_stack.check(JsonPartialValue::TYPE_PARTIAL_ARR, ',')) {
                            // Merge with partial array from previous stack.
                            auto *prev_arr = static_cast<JsonPartialArrayHead *>(main_stack.get(2)->partial_array);
                            prev_arr->final->next = value->partial_array;
                            prev_arr->final = static_cast<JsonPartialArrayHead *>(value->partial_array)->final;
                            main_stack.pop(1);
                        } else {
                            main_stack.push(value);
                        }
                        break;
                    case JsonPartialValue::TYPE_CHAR:
                        switch (char ch = value->structural) {
                            case '{':
                            case '[':
                            case ':':
                                main_stack.push(ch);
                                break;

                                // Perform reduce on the stack.
                                // There will be at most two consecutive reduce ops:
                                //   1. From a partial array to array, or from a partial object to object.
                                //   2. Merge the previously constructed value with a partial array or partial object.
                            case '}':
                                // "{", partial-object, "}"  =>  object
                                main_stack.reduce_object();
                                break;
                            case ']':
                                // "[", partial-array, "]"  =>  array
                                main_stack.reduce_array();
                                break;
                            case ',':
                                // value, ","  =>  partial-array
                                main_stack.reduce_partial_array();
                                break;
                            default:
                                error("JSON value");
                        }
                        break;
                }
                while (main_stack.reduce_partial_object()) {}
            }
        }
#else
        ParseStack main_stack(allocator);
        _thread_shift_reduce_parsing(indices, indices + num_indices - 1, &main_stack);
#endif
//        main_stack.print();
        assert(main_stack.size() == 1);
        auto *ret = reinterpret_cast<JsonValue *>(main_stack[0]);
        idx_ptr += num_indices - 1;  // Consume the indices to satisfy null ending check.

        return ret;
    }

#undef error
}
