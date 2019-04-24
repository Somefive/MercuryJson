#include <chrono>

#include <cstdio>
#include <cstring>
#include <immintrin.h>

#include "mercuryparser.h"
#include "tests.h"
#include "utils.h"

#include "linux-perf-events.h"
#include <vector>

#ifndef FORCEONEITERATION
#define FORCEONEITERATION 0
#endif

#ifndef PRINTJSON
#define PRINTJSON 0
#endif

int main(int argc, char **argv) {

    std::vector<int> evts;
    evts.push_back(PERF_COUNT_HW_CPU_CYCLES);
    evts.push_back(PERF_COUNT_HW_INSTRUCTIONS);
    evts.push_back(PERF_COUNT_HW_BRANCH_MISSES);
    evts.push_back(PERF_COUNT_HW_CACHE_REFERENCES);
    evts.push_back(PERF_COUNT_HW_CACHE_MISSES);
    LinuxEvents<PERF_TYPE_HARDWARE> unified(evts);
    std::vector<unsigned long long> results;
    results.resize(evts.size());
    unsigned long cy0 = 0, cy1 = 0, cy2 = 0;
    unsigned long cl0 = 0, cl1 = 0, cl2 = 0;
    unsigned long mis0 = 0, mis1 = 0, mis2 = 0;
    unsigned long cref0 = 0, cref1 = 0, cref2 = 0;
    unsigned long cmis0 = 0, cmis1 = 0, cmis2 = 0;

//    test_extract_mask();
//    test_extract_warp_mask();
//    test_tfn_value();

//    test_parse(true);
//    test_parse_str_naive();
//    test_parse_str_avx();
//    test_parse_str_per_bit();
//    test_parse_string();
//    test_parse_float();
//    test_translate();

//    test_remove_escaper();

    if (argc > 1) {

        size_t size;
        char *buf = read_file(argv[1], &size);
        printf("File size: %lu\n", size);

        double total_time = 0.0, best_time = 1e10, total_stage1_time = 0.0, total_stage2_time = 0.0;
        size_t iterations = FORCEONEITERATION ? 1 : (size < 1 * 1000 * 1000 ? 1000 : 10);
      
        for (size_t i = 0; i < iterations; ++i) {
            
            unified.start();
            char *input = (char *)aligned_malloc(ALIGNMENT_SIZE, size + 2 * ALIGNMENT_SIZE);
            memcpy(input, buf, size);
            auto json = MercuryJson::JSON(input, size, true);
            unified.end(results);
            cy0 += results[0];
            cl0 += results[1];
            mis0 += results[2];
            cref0 += results[3];
            cmis0 += results[4];
            
            
            auto start_time = std::chrono::steady_clock::now();
            unified.start();
            json.exec_stage1();
            unified.end(results);
            cy1 += results[0];
            cl1 += results[1];
            mis1 += results[2];
            cref1 += results[3];
            cmis1 += results[4];
            auto stage1_end_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> stage1_time = stage1_end_time - start_time;

            unified.start();
            json.exec_stage2();
            unified.end(results);
            cy2 += results[0];
            cl2 += results[1];
            mis2 += results[2];
            cref2 += results[3];
            cmis2 += results[4];
            std::chrono::duration<double> stage2_time = std::chrono::steady_clock::now() - stage1_end_time;
            double runtime = stage1_time.count() + stage2_time.count();
            
            total_time += runtime;
            best_time = std::min(best_time, runtime);
            total_stage1_time += stage1_time.count();
            total_stage2_time += stage2_time.count();

            if (PRINTJSON && i == iterations - 1) print_json(json.document);

            unified.start();
            aligned_free(input);
            unified.end(results);
            cy0 += results[0];
            cl0 += results[1];
            mis0 += results[2];
            cref0 += results[3];
            cmis0 += results[4];
        }

        unsigned long total = cy0 + cy1 + cy2;
        printf("> mem alloc/free instructions: %10lu \n\tcycles: \t\t%10lu (%.2f %%) \n\tins/cycles: "
            "\t\t%10.2f \n\tmis. branches: \t\t%10lu (cycles/mis.branch %.2f) \n\tcache accesses: "
            "\t%10lu (failure %10lu)\n",
            cl0 / iterations, cy0 / iterations, 100. * cy0 / total,
            (double)cl0 / cy0, mis0 / iterations, (double)cy0 / mis0,
            cref1 / iterations, cmis0 / iterations);
        printf("< mem alloc/free runs at %.2f cycles per input byte.\n\n",
            (double)cy0 / (iterations * size));
        printf("> stage 1 instructions: %10lu \n\tcycles: \t\t%10lu (%.2f %%) \n\tins/cycles: "
            "\t\t%10.2f \n\tmis. branches: \t\t%10lu (cycles/mis.branch %.2f) \n\tcache accesses: "
            "\t%10lu (failure %10lu)\n",
            cl1 / iterations, cy1 / iterations, 100. * cy1 / total,
            (double)cl1 / cy1, mis1 / iterations, (double)cy1 / mis1,
            cref1 / iterations, cmis1 / iterations);
        printf("< stage 1 runs at %.2f cycles per input byte.\n\n",
            (double)cy1 / (iterations * size));

        printf("> stage 2 instructions: %10lu \n\tcycles: \t\t%10lu (%.2f %%) \n\tins/cycles: "
            "\t\t%10.2f \n\tmis. branches: \t\t%10lu (cycles/mis.branch %.2f)  \n\tcache "
            "accesses: \t%10lu (failure %10lu)\n",
            cl2 / iterations, cy2 / iterations, 100. * cy2 / total,
            (double)cl2 / cy2, mis2 / iterations, (double)cy2 / mis2,
            cref2 / iterations, cmis2 / iterations);
        printf("< stage 2 runs at %.2f cycles per input byte.\n\n",
            (double)cy2 / (iterations * size));
        printf("=== all stages: %.2f cycles per input byte. ===\n",
            (double)total / (iterations * size));


        printf("Average runtime: %.6lf s, speed: %.2lf MB/s\n",
               total_time / iterations, (size * iterations / 1024.0 / 1024.0) / total_time);
        printf("Average stage 1 runtime: %.6lf s, stage 2 runtime: %.6lf s\n",
               total_stage1_time / iterations, total_stage2_time / iterations);
        printf("Best runtime: %.6lf s, speed: %.2lf MB/s\n",
               best_time, (size / 1024.0 / 1024.0) / best_time);
    }
    return 0;
}
