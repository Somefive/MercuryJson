#include <algorithm>
#include <chrono>

#include "rapidjson/include/rapidjson/document.h"
#include "rapidjson/include/rapidjson/reader.h"
#include "rapidjson/include/rapidjson/stringbuffer.h"
#include "rapidjson/include/rapidjson/writer.h"
#include "../src/utils.h"

using namespace rapidjson;


const size_t REPEAT = 1000;

int main(int argc, char **argv) {
    size_t size;
    char *buf = read_file(argv[1], &size);
    printf("File size: %lu\n", size);

    double total_time = 0.0, best_time = 1e10;
    for (size_t i = 0; i < REPEAT; ++i) {
        char *input = (char *)aligned_malloc(size + 2 * ALIGNMENT_SIZE);
        memcpy(input, buf, size);
        rapidjson::Document d;
        auto start_time = std::chrono::steady_clock::now();
        // d.Parse<rapidjson::kParseValidateEncodingFlag>(input);
        d.ParseInsitu<rapidjson::kParseValidateEncodingFlag>(input);
        if (d.HasParseError()) printf("Error\n");
        std::chrono::duration<double> runtime = std::chrono::steady_clock::now() - start_time;

        total_time += runtime.count();
        best_time = std::min(best_time, runtime.count());

        aligned_free(input);
    }
    printf("Average runtime: %.4lf s, speed: %.2lf MB/s\n",
           total_time / REPEAT, (size * REPEAT / 1024.0 / 1024.0) / total_time);
    printf("Best runtime: %.4lf s, speed: %.2lf MB/s\n",
           best_time, (size / 1024.0 / 1024.0) / best_time);
}
