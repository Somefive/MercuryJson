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
        double total_time = 0, total_s1_time = 0, total_s2_time = 0;
        clock_t mid;
        for (size_t i = 0; i < REPEAT; ++i) {
            size_t size;
            char *buf = read_file(argv[1], &size);
            auto start_time = clock();
            auto json = MercuryJson::parseJson(buf, size, mid);
            auto end_time = clock();
            auto s1_runtime = static_cast<double>(mid - start_time) / CLOCKS_PER_SEC;
            auto s2_runtime = static_cast<double>(end_time - mid) / CLOCKS_PER_SEC;
            auto runtime = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
            total_size += size;
            total_s1_time += s1_runtime;
            total_s2_time += s2_runtime;
            total_time += runtime;
            if (i == REPEAT - 1 && size < 1000) print_json(json);
            aligned_free(buf);
        }
        printf("Total Runtime: %.4lf s\n", total_time);
        printf("Average Runtime: %.4lf s\n", total_time / REPEAT);
        printf("Stage1 Runtime: %.4lf s\nStage2 Runtime: %.4lf s\n", total_s1_time, total_s2_time);
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
