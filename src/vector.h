#ifndef SQLITE_HNSW_VECTOR_H
#define SQLITE_HNSW_VECTOR_H

#include <stddef.h>
#include <stdint.h>

#include "sqlite3ext.h"

#define HNSW_ALLOCATION_LIMIT ((sqlite3_uint64)0x7fffffffU)

enum HnswMetric {
  HNSW_METRIC_L2 = 0,
  HNSW_METRIC_COSINE = 1,
  HNSW_METRIC_INNER_PRODUCT = 2
};

int hnsw_vector_validate(const void *blob, int bytes, int expected_dims,
                         int metric, char **error);
int hnsw_vector_decode(const void *blob, int bytes, float **out, int *dims,
                       char **error);
int hnsw_vector_decode_into(const void *blob, int bytes, float *out, int dims,
                            char **error);
unsigned char *hnsw_vector_encode(const float *values, int dims);
double hnsw_vector_norm(const float *values, int dims);
double hnsw_vector_distance(const float *a, double a_norm, const float *b,
                            double b_norm, int dims, int metric);
float hnsw_vector_distance_fast(const float *a, double a_norm, const float *b,
                                double b_norm, int dims, int metric);

void hnsw_sql_f32(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void hnsw_sql_dims(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void hnsw_sql_distance_l2(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void hnsw_sql_distance_cosine(sqlite3_context *ctx, int argc,
                              sqlite3_value **argv);
void hnsw_sql_distance_ip(sqlite3_context *ctx, int argc, sqlite3_value **argv);

#endif
