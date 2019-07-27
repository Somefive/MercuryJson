# MercuryJson: Multi-Threaded JSON Parsing with SIMD

This repository contains the source code for our course project in CMU 15-618: Parallel Computer Architecture and Programming.

The project is still a proof-of-concept. The currently supported functions are parsing/validating and pretty-printing. We plan to add an iterator interface soon. 

## (Brief) Introduction

MercuryJson is a fast JSON parser optimized for parsing very large documents. The idea is based mainly on the two-stage parsing framework of [simdjson](https://github.com/lemire/simdjson). Our main contribution is that we parallelized the second stage using multi-threading.

Benchmarks show that we achieve considerable speedup on large (> 500MB) documents and comparable performance on most small (< 3MB) documents.

For a detailed description of the algorithms and benchmarks, please refer to our [report](https://github.com/Somefive/MercuryJson/blob/master/report/report.pdf).

## Installation

To build MercuryJson, you will need:

- CMake version 3.0 and after.
- C++ compiler supporting the C++17 standard.
- Linux or macOS. Windows is not yet supported.
- An Intel CPU supporting the AVX2 instruction set.

Building commands are:

```bash
git clone https://github.com/Somefive/MercuryJson
cd MercuryJson
mkdir build && cd build
cmake ..
make
```

This will generate a binary named `main` under the `build` directory. This program is used for benchmarking: it reports timing for parsing of the given document. Here is an example output:

```
$ ./main ../data/large/citylots.json
File size: 189778220
Structural characters: 33395428
Iteration 0: stage 1 runtime: 0.197731 s, stage 2 runtime: 0.179965 s
Iteration 1: stage 1 runtime: 0.208775 s, stage 2 runtime: 0.173601 s
Iteration 2: stage 1 runtime: 0.196363 s, stage 2 runtime: 0.171210 s
Iteration 3: stage 1 runtime: 0.199372 s, stage 2 runtime: 0.171221 s
Iteration 4: stage 1 runtime: 0.207167 s, stage 2 runtime: 0.173756 s
Iteration 5: stage 1 runtime: 0.194635 s, stage 2 runtime: 0.173653 s
Iteration 6: stage 1 runtime: 0.208866 s, stage 2 runtime: 0.175466 s
Iteration 7: stage 1 runtime: 0.196667 s, stage 2 runtime: 0.169261 s
Iteration 8: stage 1 runtime: 0.193073 s, stage 2 runtime: 0.171227 s
Iteration 9: stage 1 runtime: 0.192630 s, stage 2 runtime: 0.168801 s
Average runtime: 0.372344 s, speed: 486.07 MB/s
Average stage 1 runtime: 0.199528 s (53.59 %), stage 2 runtime: 0.172816 s (46.41 %)
Best runtime: 0.361431 s, speed: 500.75 MB/s
```

All configurable flags are stored in `src/flags.h`. Note that the number of threads to use is hardcoded at compile time.

## Caveats

The following features are not yet supported by our parser:

- Null characters (`'\0'`) within strings; currently we use null-terminated C-style strings.
- Conversion & validation of escaped Unicode characters.
- Comments (`/**/`).

The following incorrect JSON fragments are accepted by our parser:

- Unescaped control characters within strings.
- Invalid escape sequences.
- Escaped characters outside strings.

For detailed discussion on JSON standards, please see [JSON Test Suite](https://github.com/nst/JSONTestSuite).
