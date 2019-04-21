#include <cstdio>
#include <immintrin.h>
#include <chrono>

#include "tests.h"
#include "utils.h"
#include "mercuryparser.h"

int main(int argc, char **argv) {
    if (argc > 1) {
        size_t size;
        char *buf = read_file(argv[1], &size);
        auto start_time = clock();
        auto json = MercuryJson::parseJson(buf, size);
        auto runtime = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
        printf("Runtime: %.4lf s\n", runtime);
        printf("Speed: %.2lf MB/s\n", (size / 1024.0 / 1024.0) / runtime);
        if (size < 1000) print_json(json);
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
