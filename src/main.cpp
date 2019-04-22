#include <cstdio>
#include <immintrin.h>
#include <chrono>

#include "tests.h"
#include "utils.h"
#include "mercuryparser.h"

const int REPEAT = 1000;

int main(int argc, char **argv) {
    if (argc > 1) {
        double total_time = 0, best_time = 1e10;
        size_t size;
        char *buf = read_file(argv[1], &size);
        printf("File size: %lu\n", size);
        for (size_t i = 0; i < REPEAT; ++i) {
            char *input = (char *)aligned_malloc(ALIGNMENT_SIZE, (size / ALIGNMENT_SIZE + 2) * ALIGNMENT_SIZE);
            memcpy(input, buf, size);
            auto start_time = std::chrono::steady_clock::now();
            auto json = MercuryJson::JSON(input, size);
            std::chrono::duration<double> runtime = std::chrono::steady_clock::now() - start_time;
            total_time += runtime.count();
            best_time = std::min(best_time, runtime.count());
            if (i == REPEAT - 1 && size < 1000) print_json(json.document);
            aligned_free(input);
        }
        printf("Average runtime: %.4lf s, speed: %.2lf MB/s\n",
                total_time / REPEAT, (size * REPEAT / 1024.0 / 1024.0) / total_time);
        printf("Best runtime: %.4lf s, speed: %.2lf MB/s\n", best_time, (size / 1024.0 / 1024.0) / best_time);
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
