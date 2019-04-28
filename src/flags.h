#ifndef MERCURYJSON_FLAGS_H
#define MERCURYJSON_FLAGS_H

// Whether to use static variables in __cmpeq_mask.
#ifndef STATIC_CMPEQ_MASK
# define STATIC_CMPEQ_MASK 0
#endif

// Mode for string parsing. 0 for naive, 1 for AVX, 2 for per_bit (simdjson-style).
#ifndef PARSE_STR_MODE
# define PARSE_STR_MODE 1
#endif

// Whether to allocate new memory for parsed strings. Set to 0 to do in-place parsing.
#if PARSE_STR_MODE == 2
# define ALLOC_PARSED_STR 1
#else
# ifndef ALLOC_PARSED_STR
#  define ALLOC_PARSED_STR 0
# endif
#endif

// Whether to use stack-based shift-reduce parsing
#ifndef SHIFT_REDUCE_PARSER
# define SHIFT_REDUCE_PARSER 1
#endif


// Whether to use fully-vectorized string parsing implementation. Only works when PARSE_STR_MODE == 1.
#ifndef PARSE_STR_FULLY_AXV
# define PARSE_STR_FULLY_AXV 0
#endif

// Whether to enable vectorized operations for number parsing.
#ifndef PARSE_NUMBER_AVX
# define PARSE_NUMBER_AVX 1
#endif

// Whether to perform string parsing on a dedicated thread.
#ifndef PARSE_STR_MULTITHREAD
# define PARSE_STR_MULTITHREAD 1
#endif

// Whether to use block allocator for dynamic memory allocation.
#ifndef USE_BLOCK_ALLOCATOR
# define USE_BLOCK_ALLOCATOR 1
#endif

#endif // MERCURYJSON_FLAGS_H
