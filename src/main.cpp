#include <chrono>
#include <iomanip>
#include <vector>

#include <immintrin.h>
#include <stdio.h>
#include <string.h>

#include "flags.h"
#include "mercuryparser.h"
#include "tape.h"
#include "tests.h"
#include "utils.h"

#if PERF_EVENTS
# include "linux-perf-events.h"
#endif

void run(int argc, char **argv) {
#if PERF_EVENTS
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
#endif

    if (argc > 1) {

        std::cout << std::fixed << std::setprecision(10);

        size_t size;
        char *buf = read_file(argv[1], &size);
        printf("File size: %lu\n", size);
        char *input = aligned_malloc(size + 2 * kAlignmentSize);

        double total_time = 0.0, best_time = 1e10, total_stage1_time = 0.0, total_stage2_time = 0.0;
        size_t iterations = FORCE_ONE_ITERATION ? 1 : (size < 100 * 1000 * 1000 ? 1000 : 10);

        for (size_t i = 0; i < iterations; ++i) {
#if PERF_EVENTS
            unified.start();
#endif
            memcpy(input, buf, size);
            auto json = MercuryJson::JSON(input, size, true);
#if USE_TAPE
            MercuryJson::Tape tape(size, size);
#endif
#if PERF_EVENTS
            unified.end(results);
            cy0 += results[0];
            cl0 += results[1];
            mis0 += results[2];
            cref0 += results[3];
            cmis0 += results[4];
#endif

            auto start_time = std::chrono::steady_clock::now();
#if PERF_EVENTS
            unified.start();
#endif
            json.exec_stage1();
            if (i == 0) printf("Structural characters: %lu\n", json.num_indices);
#if PERF_EVENTS
            unified.end(results);
            cy1 += results[0];
            cl1 += results[1];
            mis1 += results[2];
            cref1 += results[3];
            cmis1 += results[4];
#endif
            auto stage1_end_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> stage1_time = stage1_end_time - start_time;

#if PERF_EVENTS
            unified.start();
#endif
#if USE_TAPE
#if TAPE_STATE_MACHINE
            tape.state_machine(const_cast<char *>(json.input), json.indices, json.num_indices);
#else
            MercuryJson::TapeWriter tape_writer(&tape, json.input, json.indices);
            tape_writer.parse_value();
#endif
#else
            json.exec_stage2();
#endif
#if PERF_EVENTS
            unified.end(results);
            cy2 += results[0];
            cl2 += results[1];
            mis2 += results[2];
            cref2 += results[3];
            cmis2 += results[4];
#endif
            std::chrono::duration<double> stage2_time = std::chrono::steady_clock::now() - stage1_end_time;
            double runtime = stage1_time.count() + stage2_time.count();

            total_time += runtime;
            best_time = std::min(best_time, runtime);
            total_stage1_time += stage1_time.count();
            total_stage2_time += stage2_time.count();

            if (iterations <= 10) {
                printf("Iteration %lu: stage 1 runtime: %.6lf s, stage 2 runtime: %.6lf s\n",
                       i, stage1_time.count(), stage2_time.count());
            }

#if PRINT_JSON
            if (i == iterations - 1) {
# if USE_TAPE
                tape.print_json();
# else
                print_json(json.document);
# endif
            }
#endif
#if PERF_EVENTS
            unified.start();
            unified.end(results);
            cy0 += results[0];
            cl0 += results[1];
            mis0 += results[2];
            cref0 += results[3];
            cmis0 += results[4];
#endif
        }

        aligned_free(input);

#if PERF_EVENTS
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
#endif

        printf("Average runtime: %.6lf s, speed: %.2lf MB/s\n",
               total_time / iterations, (size * iterations / 1024.0 / 1024.0) / total_time);
        printf("Average stage 1 runtime: %.6lf s, stage 2 runtime: %.6lf s\n",
               total_stage1_time / iterations, total_stage2_time / iterations);
        printf("Best runtime: %.6lf s, speed: %.2lf MB/s\n",
               best_time, (size / 1024.0 / 1024.0) / best_time);
    }
}

int main(int argc, char **argv) {

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

//    if (argc > 1) {
//        test_tape(argv[1]);
//    }

   run(argc, argv);

    return 0;
}
