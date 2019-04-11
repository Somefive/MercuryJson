#include <cstdio>
#include "utils.h"
#include <immintrin.h>
#include "tests.hpp"

int main() {
    test_extract_mask();
    // const char * buf = read_file("data/demographic_statistics_by_zipcode.json");
    test_extract_warp_mask();
    return 0;
}