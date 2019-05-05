#ifndef MERCURYJSON_FLAGS_H
#define MERCURYJSON_FLAGS_H

/* String Parsing */
// Mode for string parsing. -1 for disable, 0 for naive, 1 for AVX, 2 for per_bit (simdjson-style).
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

// Whether to use fully-vectorized string parsing implementation. Only works when PARSE_STR_MODE == 1.
#ifndef PARSE_STR_FULLY_AVX
# define PARSE_STR_FULLY_AVX 0
#endif

// Whether to use 32-bit blocks for string parsing. Only works when PARSE_STR_MODE == 1 and PARSE_STR_FULLY_AVX == 0.
#if PARSE_STR_FULLY_AVX
# define PARSE_STR_32BIT 0
#else
# ifndef PARSE_STR_32BIT
#  define PARSE_STR_32BIT 0
# endif
#endif

// Number of extra dedicated threads for string parsing. Set to 0 to disable.
#ifndef PARSE_STR_NUM_THREADS
# define PARSE_STR_NUM_THREADS 4
#endif


/* Number Parsing */
// Whether to parse numbers. Set to 0 only for performance tests.
#ifndef NO_PARSE_NUMBER
# define NO_PARSE_NUMBER 0
#endif

// Whether to enable vectorized operations for number parsing.
#ifndef PARSE_NUMBER_AVX
# define PARSE_NUMBER_AVX 1
#endif

// Number of extra dedicated threads for number parsing. Set to 0 to disable.
#ifndef PARSE_NUM_NUM_THREADS
# define PARSE_NUM_NUM_THREADS 0
#endif


/* Main Parsing Algorithm */
// Whether to use stack-based shift-reduce parsing.
#ifndef SHIFT_REDUCE_PARSER
# define SHIFT_REDUCE_PARSER 0
#endif

// Number of threads to use for shift-reduce parsing.
#ifndef SHIFT_REDUCE_NUM_THREADS
# define SHIFT_REDUCE_NUM_THREADS 4
#endif

// Whether to use tape data structure for parsing.
#ifndef USE_TAPE
# define USE_TAPE 1
#endif

// Whether to use state machine-based tape parsing.
#ifndef TAPE_STATE_MACHINE
# define TAPE_STATE_MACHINE 1
#endif

// Number of threads to use for state machine-based tape parsing.
#ifndef TAPE_STATE_MACHINE_NUM_THREADS
# define TAPE_STATE_MACHINE_NUM_THREADS 4
#endif


/* Testing */
// Whether to run performance test for only one iteration.
#ifndef FORCE_ONE_ITERATION
# define FORCE_ONE_ITERATION 0
#endif

// Whether to print the parsed JSON.
#ifndef PRINT_JSON
# define PRINT_JSON 0
#endif

// Whether to perf events. Only available for Linux.
#ifdef __linux__
# define PERF_EVENTS 1
#else
# define PERF_EVENTS 0
#endif

#endif // MERCURYJSON_FLAGS_H
