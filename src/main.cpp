#include <cstdio>
#include <immintrin.h>
#include <chrono>

#include "tests.h"
#include "utils.h"
#include "mercuryparser.h"

#define REPEAT 1000

int main(int argc, char **argv) {
    if (argc > 1) {
        size_t total_size = 0;
        double total_time = 0;
        for (size_t i = 0; i < REPEAT; ++i) {
            size_t size;
            char *buf = read_file(argv[1], &size);
            auto start_time = clock();
            auto json = MercuryJson::parseJson(buf, size);
            auto runtime = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
            total_size += size;
            total_time += runtime;
            if (i == REPEAT - 1 && size < 1000) print_json(json);
            aligned_free(buf);
        }
        printf("Average Runtime: %.4lf s\n", total_time / REPEAT);
        printf("Speed: %.2lf MB/s\n", (total_size / 1024.0 / 1024.0) / total_time);
    }

//    test_extract_mask();
//    test_extract_warp_mask();
//    test_tfn_value();

//    test_parse(true);
//    test_parseStr();
//    test_parseStrAVX();
//    test_parseString();
//    test_translate();

//    test_remove_escaper();
    return 0;
}
