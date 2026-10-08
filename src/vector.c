#include "vector.h"
SQLITE_EXTENSION_INIT3

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__)) &&                              \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define HNSW_HAVE_X86_AVX2 1
#else
#define HNSW_HAVE_X86_AVX2 0
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#define HNSW_HAVE_ARM_NEON 1
#else
#define HNSW_HAVE_ARM_NEON 0
#endif

static uint32_t read_u32_le(const unsigned char *p) {
  return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8U) | ((uint32_t)p[2] << 16U) |
         ((uint32_t)p[3] << 24U);
}

static void write_u32_le(unsigned char *p, uint32_t value) {
  p[0] = (unsigned char)(value & 0xffU);
  p[1] = (unsigned char)((value >> 8U) & 0xffU);
  p[2] = (unsigned char)((value >> 16U) & 0xffU);
  p[3] = (unsigned char)((value >> 24U) & 0xffU);
}

static float read_f32_le(const unsigned char *p) {
  uint32_t bits = read_u32_le(p);
  float value = 0.0F;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static void write_f32_le(unsigned char *p, float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  write_u32_le(p, bits);
}

int hnsw_vector_validate(const void *blob, int bytes, int expected_dims,
                         int metric, char **error) {
  const unsigned char *data = (const unsigned char *)blob;
  int dims = 0;
  double sum = 0.0;
  int i = 0;

  if (blob == NULL || bytes <= 0 || (bytes % 4) != 0) {
    if (error != NULL) {
      *error = sqlite3_mprintf("vector must be a non-empty float32 BLOB");
    }
    return SQLITE_MISMATCH;
  }
  dims = bytes / 4;
  if (expected_dims > 0 && dims != expected_dims) {
    if (error != NULL) {
      *error = sqlite3_mprintf("vector has %d dimensions; expected %d", dims,
                               expected_dims);
    }
    return SQLITE_CONSTRAINT;
  }
  for (i = 0; i < dims; ++i) {
    float value = read_f32_le(data + (size_t)i * 4U);
    if (!isfinite(value)) {
      if (error != NULL) {
        *error = sqlite3_mprintf("vector contains a non-finite value at %d", i);
      }
      return SQLITE_CONSTRAINT;
    }
    sum += (double)value * (double)value;
  }
  if (metric == HNSW_METRIC_COSINE && sum == 0.0) {
    if (error != NULL) {
      *error = sqlite3_mprintf("cosine vectors must have non-zero norm");
    }
    return SQLITE_CONSTRAINT;
  }
  return SQLITE_OK;
}

int hnsw_vector_decode(const void *blob, int bytes, float **out, int *dims,
                       char **error) {
  const unsigned char *data = (const unsigned char *)blob;
  float *values = NULL;
  int count = 0;
  int i = 0;
  int rc = hnsw_vector_validate(blob, bytes, 0, HNSW_METRIC_L2, error);
  if (rc != SQLITE_OK) {
    return rc;
  }
  count = bytes / 4;
  if ((sqlite3_uint64)count > HNSW_ALLOCATION_LIMIT / sizeof(float)) {
    if (error != NULL) {
      *error = sqlite3_mprintf("vector is too large");
    }
    return SQLITE_TOOBIG;
  }
  values = sqlite3_malloc64((sqlite3_uint64)count * sizeof(float));
  if (values == NULL) {
    return SQLITE_NOMEM;
  }
  for (i = 0; i < count; ++i) {
    values[i] = read_f32_le(data + (size_t)i * 4U);
  }
  *out = values;
  *dims = count;
  return SQLITE_OK;
}

int hnsw_vector_decode_into(const void *blob, int bytes, float *out, int dims,
                            char **error) {
  const unsigned char *data = (const unsigned char *)blob;
  int i = 0;
  int rc = hnsw_vector_validate(blob, bytes, dims, HNSW_METRIC_L2, error);
  if (rc != SQLITE_OK) {
    return rc;
  }
  for (i = 0; i < dims; ++i) {
    out[i] = read_f32_le(data + (size_t)i * 4U);
  }
  return SQLITE_OK;
}

unsigned char *hnsw_vector_encode(const float *values, int dims) {
  unsigned char *blob = NULL;
  int i = 0;
  if (dims <= 0 || (sqlite3_uint64)dims > HNSW_ALLOCATION_LIMIT / 4U) {
    return NULL;
  }
  blob = sqlite3_malloc64((sqlite3_uint64)dims * 4U);
  if (blob == NULL) {
    return NULL;
  }
  for (i = 0; i < dims; ++i) {
    write_f32_le(blob + (size_t)i * 4U, values[i]);
  }
  return blob;
}

double hnsw_vector_norm(const float *values, int dims) {
  double sum = 0.0;
  int i = 0;
  for (i = 0; i < dims; ++i) {
    sum += (double)values[i] * (double)values[i];
  }
  return sqrt(sum);
}

#if HNSW_HAVE_X86_AVX2
__attribute__((target("avx2"))) static double
dot_avx2(const float *a, const float *b, int dims) {
  __m256d sum = _mm256_setzero_pd();
  double lanes[4];
  double result = 0.0;
  int i = 0;
  for (; i + 4 <= dims; i += 4) {
    __m128 af = _mm_loadu_ps(a + i);
    __m128 bf = _mm_loadu_ps(b + i);
    __m256d ad = _mm256_cvtps_pd(af);
    __m256d bd = _mm256_cvtps_pd(bf);
    sum = _mm256_add_pd(sum, _mm256_mul_pd(ad, bd));
  }
  _mm256_storeu_pd(lanes, sum);
  result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
  for (; i < dims; ++i) {
    result += (double)a[i] * (double)b[i];
  }
  return result;
}

__attribute__((target("avx2"))) static double
l2_avx2(const float *a, const float *b, int dims) {
  __m256d sum = _mm256_setzero_pd();
  double lanes[4];
  double result = 0.0;
  int i = 0;
  for (; i + 4 <= dims; i += 4) {
    __m128 af = _mm_loadu_ps(a + i);
    __m128 bf = _mm_loadu_ps(b + i);
    __m256d delta = _mm256_sub_pd(_mm256_cvtps_pd(af), _mm256_cvtps_pd(bf));
    sum = _mm256_add_pd(sum, _mm256_mul_pd(delta, delta));
  }
  _mm256_storeu_pd(lanes, sum);
  result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
  for (; i < dims; ++i) {
    double delta = (double)a[i] - (double)b[i];
    result += delta * delta;
  }
  return result;
}

static int use_avx2(void) { return __builtin_cpu_supports("avx2") != 0; }

__attribute__((target("avx2"))) static float
distance_fast_avx2(const float *a, double a_norm, const float *b, double b_norm,
                   int dims, int metric) {
  __m256 sum = _mm256_setzero_ps();
  float lanes[8];
  float value = 0.0F;
  int i = 0;
  for (; i + 8 <= dims; i += 8) {
    __m256 av = _mm256_loadu_ps(a + i);
    __m256 bv = _mm256_loadu_ps(b + i);
    if (metric == HNSW_METRIC_L2) {
      __m256 delta = _mm256_sub_ps(av, bv);
      sum = _mm256_add_ps(sum, _mm256_mul_ps(delta, delta));
    } else {
      sum = _mm256_add_ps(sum, _mm256_mul_ps(av, bv));
    }
  }
  _mm256_storeu_ps(lanes, sum);
  for (int lane = 0; lane < 8; ++lane) {
    value += lanes[lane];
  }
  for (; i < dims; ++i) {
    if (metric == HNSW_METRIC_L2) {
      float delta = a[i] - b[i];
      value += delta * delta;
    } else {
      value += a[i] * b[i];
    }
  }
  if (metric == HNSW_METRIC_INNER_PRODUCT) {
    return -value;
  }
  if (metric == HNSW_METRIC_COSINE) {
    return 1.0F - value / (float)(a_norm * b_norm);
  }
  return value;
}
#endif

#if HNSW_HAVE_ARM_NEON
static double dot_neon(const float *a, const float *b, int dims) {
  float64x2_t sum_low = vdupq_n_f64(0.0);
  float64x2_t sum_high = vdupq_n_f64(0.0);
  double result = 0.0;
  int i = 0;
  for (; i + 4 <= dims; i += 4) {
    float32x4_t af = vld1q_f32(a + i);
    float32x4_t bf = vld1q_f32(b + i);
    sum_low = vfmaq_f64(sum_low, vcvt_f64_f32(vget_low_f32(af)),
                        vcvt_f64_f32(vget_low_f32(bf)));
    sum_high =
        vfmaq_f64(sum_high, vcvt_high_f64_f32(af), vcvt_high_f64_f32(bf));
  }
  result = vaddvq_f64(sum_low) + vaddvq_f64(sum_high);
  for (; i < dims; ++i) {
    result += (double)a[i] * (double)b[i];
  }
  return result;
}

static double l2_neon(const float *a, const float *b, int dims) {
  float64x2_t sum_low = vdupq_n_f64(0.0);
  float64x2_t sum_high = vdupq_n_f64(0.0);
  double result = 0.0;
  int i = 0;
  for (; i + 4 <= dims; i += 4) {
    float32x4_t delta = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    float64x2_t low = vcvt_f64_f32(vget_low_f32(delta));
    float64x2_t high = vcvt_high_f64_f32(delta);
    sum_low = vfmaq_f64(sum_low, low, low);
    sum_high = vfmaq_f64(sum_high, high, high);
  }
  result = vaddvq_f64(sum_low) + vaddvq_f64(sum_high);
  for (; i < dims; ++i) {
    double delta = (double)a[i] - (double)b[i];
    result += delta * delta;
  }
  return result;
}
#endif

double hnsw_vector_distance(const float *a, double a_norm, const float *b,
                            double b_norm, int dims, int metric) {
  double value = 0.0;
  int i = 0;
  if (metric == HNSW_METRIC_L2) {
#if HNSW_HAVE_X86_AVX2
    if (dims >= 8 && use_avx2()) {
      return l2_avx2(a, b, dims);
    }
#endif
#if HNSW_HAVE_ARM_NEON
    if (dims >= 8) {
      return l2_neon(a, b, dims);
    }
#endif
    for (i = 0; i < dims; ++i) {
      double delta = (double)a[i] - (double)b[i];
      value += delta * delta;
    }
    return value;
  }
#if HNSW_HAVE_X86_AVX2
  if (dims >= 8 && use_avx2()) {
    value = dot_avx2(a, b, dims);
  } else
#endif
#if HNSW_HAVE_ARM_NEON
      if (dims >= 8) {
    value = dot_neon(a, b, dims);
  } else
#endif
  {
    for (i = 0; i < dims; ++i) {
      value += (double)a[i] * (double)b[i];
    }
  }
  if (metric == HNSW_METRIC_INNER_PRODUCT) {
    return -value;
  }
  if (a_norm == 0.0 || b_norm == 0.0) {
    return INFINITY;
  }
  return 1.0 - (value / (a_norm * b_norm));
}

float hnsw_vector_distance_fast(const float *a, double a_norm, const float *b,
                                double b_norm, int dims, int metric) {
  float value = 0.0F;
  int i = 0;
#if HNSW_HAVE_X86_AVX2
  if (dims >= 8 && use_avx2()) {
    return distance_fast_avx2(a, a_norm, b, b_norm, dims, metric);
  }
#elif HNSW_HAVE_ARM_NEON
  if (dims >= 4) {
    float32x4_t sum = vdupq_n_f32(0.0F);
    for (; i + 4 <= dims; i += 4) {
      float32x4_t av = vld1q_f32(a + i);
      float32x4_t bv = vld1q_f32(b + i);
      if (metric == HNSW_METRIC_L2) {
        float32x4_t delta = vsubq_f32(av, bv);
        sum = vfmaq_f32(sum, delta, delta);
      } else {
        sum = vfmaq_f32(sum, av, bv);
      }
    }
    value = vaddvq_f32(sum);
  }
#endif
  for (; i < dims; ++i) {
    if (metric == HNSW_METRIC_L2) {
      float delta = a[i] - b[i];
      value += delta * delta;
    } else {
      value += a[i] * b[i];
    }
  }
  if (metric == HNSW_METRIC_INNER_PRODUCT) {
    return -value;
  }
  if (metric == HNSW_METRIC_COSINE) {
    if (a_norm == 0.0 || b_norm == 0.0) {
      return INFINITY;
    }
    return 1.0F - value / (float)(a_norm * b_norm);
  }
  return value;
}

static int parse_json_vector(const char *text, int bytes, float **out,
                             int *dims, char **error) {
  const char *p = text;
  const char *end = text + bytes;
  float *values = NULL;
  int count = 0;
  int capacity = 0;

  while (p < end && isspace((unsigned char)*p)) {
    ++p;
  }
  if (p == end || *p++ != '[') {
    *error = sqlite3_mprintf("vector text must be a JSON array");
    return SQLITE_MISMATCH;
  }
  for (;;) {
    char *number_end = NULL;
    double parsed = 0.0;
    while (p < end && isspace((unsigned char)*p)) {
      ++p;
    }
    if (p < end && *p == ']') {
      ++p;
      break;
    }
    if (p == end) {
      sqlite3_free(values);
      *error = sqlite3_mprintf("unterminated vector array");
      return SQLITE_MISMATCH;
    }
    errno = 0;
    parsed = strtod(p, &number_end);
    if (number_end == p || number_end > end || errno == ERANGE ||
        !isfinite(parsed) || parsed > (double)FLT_MAX ||
        parsed < -(double)FLT_MAX) {
      sqlite3_free(values);
      *error = sqlite3_mprintf("invalid float32 vector element");
      return SQLITE_MISMATCH;
    }
    if (count == capacity) {
      int next = capacity == 0 ? 16 : capacity * 2;
      float *grown = NULL;
      if (next < capacity ||
          (sqlite3_uint64)next > HNSW_ALLOCATION_LIMIT / sizeof(float)) {
        sqlite3_free(values);
        return SQLITE_TOOBIG;
      }
      grown = sqlite3_realloc64(values, (sqlite3_uint64)next * sizeof(float));
      if (grown == NULL) {
        sqlite3_free(values);
        return SQLITE_NOMEM;
      }
      values = grown;
      capacity = next;
    }
    values[count++] = (float)parsed;
    p = number_end;
    while (p < end && isspace((unsigned char)*p)) {
      ++p;
    }
    if (p < end && *p == ',') {
      ++p;
      while (p < end && isspace((unsigned char)*p)) {
        ++p;
      }
      if (p == end || *p == ']') {
        sqlite3_free(values);
        *error = sqlite3_mprintf("vector array has a trailing comma");
        return SQLITE_MISMATCH;
      }
      continue;
    }
    if (p < end && *p == ']') {
      ++p;
      break;
    }
    sqlite3_free(values);
    *error = sqlite3_mprintf("expected ',' or ']' in vector array");
    return SQLITE_MISMATCH;
  }
  while (p < end && isspace((unsigned char)*p)) {
    ++p;
  }
  if (p != end || count == 0) {
    sqlite3_free(values);
    *error = sqlite3_mprintf(count == 0 ? "vector must not be empty"
                                        : "unexpected text after vector");
    return SQLITE_MISMATCH;
  }
  *out = values;
  *dims = count;
  return SQLITE_OK;
}

void hnsw_sql_f32(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  int type = SQLITE_NULL;
  char *error = NULL;
  (void)argc;
  type = sqlite3_value_type(argv[0]);
  if (type == SQLITE_BLOB) {
    const void *blob = sqlite3_value_blob(argv[0]);
    int bytes = sqlite3_value_bytes(argv[0]);
    int rc = hnsw_vector_validate(blob, bytes, 0, HNSW_METRIC_L2, &error);
    if (rc != SQLITE_OK) {
      sqlite3_result_error(ctx, error != NULL ? error : "invalid vector", -1);
      sqlite3_free(error);
      return;
    }
    sqlite3_result_blob(ctx, blob, bytes, SQLITE_TRANSIENT);
    return;
  }
  if (type == SQLITE_TEXT) {
    const char *text = (const char *)sqlite3_value_text(argv[0]);
    int bytes = sqlite3_value_bytes(argv[0]);
    float *values = NULL;
    unsigned char *blob = NULL;
    int dims = 0;
    int rc = parse_json_vector(text, bytes, &values, &dims, &error);
    if (rc != SQLITE_OK) {
      if (rc == SQLITE_NOMEM) {
        sqlite3_result_error_nomem(ctx);
      } else {
        sqlite3_result_error(ctx, error != NULL ? error : "invalid vector", -1);
      }
      sqlite3_free(error);
      return;
    }
    blob = hnsw_vector_encode(values, dims);
    sqlite3_free(values);
    if (blob == NULL) {
      sqlite3_result_error_nomem(ctx);
      return;
    }
    sqlite3_result_blob(ctx, blob, dims * 4, sqlite3_free);
    return;
  }
  sqlite3_result_error(ctx, "hnsw_f32 expects JSON text or a float32 BLOB", -1);
}

void hnsw_sql_dims(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  const void *blob = NULL;
  int bytes = 0;
  char *error = NULL;
  (void)argc;
  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
    sqlite3_result_error(ctx, "hnsw_dims expects a float32 BLOB", -1);
    return;
  }
  blob = sqlite3_value_blob(argv[0]);
  bytes = sqlite3_value_bytes(argv[0]);
  if (hnsw_vector_validate(blob, bytes, 0, HNSW_METRIC_L2, &error) !=
      SQLITE_OK) {
    sqlite3_result_error(ctx, error != NULL ? error : "invalid vector", -1);
    sqlite3_free(error);
    return;
  }
  sqlite3_result_int(ctx, bytes / 4);
}

static void sql_distance(sqlite3_context *ctx, sqlite3_value **argv,
                         int metric) {
  float *a = NULL;
  float *b = NULL;
  int a_dims = 0;
  int b_dims = 0;
  char *error = NULL;
  int rc = SQLITE_OK;
  double distance = 0.0;

  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB ||
      sqlite3_value_type(argv[1]) != SQLITE_BLOB) {
    sqlite3_result_error(ctx, "distance functions expect two float32 BLOBs",
                         -1);
    return;
  }
  rc = hnsw_vector_decode(sqlite3_value_blob(argv[0]),
                          sqlite3_value_bytes(argv[0]), &a, &a_dims, &error);
  if (rc == SQLITE_OK) {
    rc = hnsw_vector_decode(sqlite3_value_blob(argv[1]),
                            sqlite3_value_bytes(argv[1]), &b, &b_dims, &error);
  }
  if (rc != SQLITE_OK || a_dims != b_dims) {
    if (rc == SQLITE_NOMEM) {
      sqlite3_result_error_nomem(ctx);
    } else if (a_dims != b_dims && rc == SQLITE_OK) {
      sqlite3_result_error(ctx, "vectors have different dimensions", -1);
    } else {
      sqlite3_result_error(ctx, error != NULL ? error : "invalid vector", -1);
    }
    sqlite3_free(error);
    sqlite3_free(a);
    sqlite3_free(b);
    return;
  }
  distance = hnsw_vector_distance(a, hnsw_vector_norm(a, a_dims), b,
                                  hnsw_vector_norm(b, b_dims), a_dims, metric);
  sqlite3_free(a);
  sqlite3_free(b);
  if (!isfinite(distance)) {
    sqlite3_result_error(ctx, "cosine vectors must have non-zero norm", -1);
    return;
  }
  if (metric == HNSW_METRIC_L2) {
    distance = sqrt(distance);
  }
  sqlite3_result_double(ctx, distance);
}

void hnsw_sql_distance_l2(sqlite3_context *ctx, int argc,
                          sqlite3_value **argv) {
  (void)argc;
  sql_distance(ctx, argv, HNSW_METRIC_L2);
}

void hnsw_sql_distance_cosine(sqlite3_context *ctx, int argc,
                              sqlite3_value **argv) {
  (void)argc;
  sql_distance(ctx, argv, HNSW_METRIC_COSINE);
}

void hnsw_sql_distance_ip(sqlite3_context *ctx, int argc,
                          sqlite3_value **argv) {
  (void)argc;
  sql_distance(ctx, argv, HNSW_METRIC_INNER_PRODUCT);
}
