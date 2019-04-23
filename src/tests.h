#ifndef MERCURYJSON_TESTS_H
#define MERCURYJSON_TESTS_H

#include "mercuryparser.h"

void test_extract_mask();
void test_extract_warp_mask();

void test_tfn_value();

void print_json(MercuryJson::JsonValue *value, int indent = 0);

void test_parse(bool print = false);

void test_parse_str_naive();
void test_parse_str_avx();
void test_parse_str_per_bit();

void test_parse_string();

void test_translate();
void test_remove_escaper();

#endif // MERCURYJSON_TESTS_H
