#include <cstdio>
#include <immintrin.h>
#include <chrono>

#include "tests.h"
#include "utils.h"
#include "mercuryparser.h"

const int REPEAT = 1000;

int main(int argc, char **argv) {
    if (argc > 1) {
        size_t size;
        char *buf = read_file(argv[1], &size);
        printf("File size: %lu\n", size);

        double total_time = 0.0, best_time = 1e10, total_stage1_time = 0.0, total_stage2_time = 0.0;
        for (size_t i = 0; i < REPEAT; ++i) {
            char *input = (char *)aligned_malloc(ALIGNMENT_SIZE, size + 2 * ALIGNMENT_SIZE);
            memcpy(input, buf, size);
            auto json = MercuryJson::JSON(input, size, true);
            auto start_time = std::chrono::steady_clock::now();
            json.exec_stage1();
            auto stage1_end_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> stage1_time = stage1_end_time - start_time;
            json.exec_stage2();
            std::chrono::duration<double> stage2_time = std::chrono::steady_clock::now() - stage1_end_time;

            double runtime = stage1_time.count() + stage2_time.count();
            total_time += runtime;
            best_time = std::min(best_time, runtime);
            total_stage1_time += stage1_time.count();
            total_stage2_time += stage2_time.count();

            if (i == REPEAT - 1 && size < 1000) print_json(json.document);
            aligned_free(input);
        }
        printf("Average runtime: %.4lf s, speed: %.2lf MB/s\n",
               total_time / REPEAT, (size * REPEAT / 1024.0 / 1024.0) / total_time);
        printf("Average stage 1 runtime: %.4lf s, stage 2 runtime: %.4lf s\n",
               total_stage1_time / REPEAT, total_stage2_time / REPEAT);
        printf("Best runtime: %.4lf s, speed: %.2lf MB/s\n",
               best_time, (size / 1024.0 / 1024.0) / best_time);
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
