#ifndef MERCURYJSON_TESTS_H
#define MERCURYJSON_TESTS_H

#include "mercuryparser.h"

void test_extract_mask();

void test_extract_warp_mask();

void test_tfn_value();

void print_json(const MercuryJson::JsonValue &value, int indent = 0);

void test_parse(bool print = false);

void test_parseStr();

void test_parseStrAVX();

void test_parseString();

void test_translate();

void test_remove_escaper();

#endif // MERCURYJSON_TESTS_H
