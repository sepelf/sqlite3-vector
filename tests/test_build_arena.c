#include "build_arena.h"
#include "vector.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float hnsw_vector_distance_fast(const float *a, double a_norm, const float *b,
                                double b_norm, int dims, int metric) {
  float value = 0.0F;
  int i = 0;
  for (i = 0; i < dims; ++i) {
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

static int initialize(HnswBuildArena *arena, int threads) {
  const size_t count = 4096;
  int rc = hnsw_build_arena_init(arena, count, 4, 8, 64, HNSW_METRIC_L2,
                                 threads, 128U * 1024U * 1024U);
  if (rc != SQLITE_OK) {
    return rc;
  }
  for (size_t i = 0; i < count; ++i) {
    arena->node_ids[i] = (sqlite3_int64)i + 1;
    arena->levels[i] =
        (unsigned char)(i % 512 == 0 ? 2 : (i % 32 == 0 ? 1 : 0));
    for (int d = 0; d < arena->dims; ++d) {
      float value = (float)((i * (size_t)(d + 3)) % 1009U) / 1009.0F;
      arena->vectors[i * (size_t)arena->dims + (size_t)d] = value;
      arena->norms[i] += (double)value * (double)value;
    }
    arena->norms[i] = sqrt(arena->norms[i]);
  }
  rc = hnsw_build_arena_prepare(arena);
  return rc == SQLITE_OK ? hnsw_build_arena_run(arena) : rc;
}

static int always_cancelled(void *context) {
  (void)context;
  return 1;
}

static int arenas_equal(const HnswBuildArena *left,
                        const HnswBuildArena *right) {
  return left->entry == right->entry && left->max_level == right->max_level &&
         left->neighbor_offsets[left->count] ==
             right->neighbor_offsets[right->count] &&
         left->count_offsets[left->count] ==
             right->count_offsets[right->count] &&
         memcmp(left->neighbor_counts, right->neighbor_counts,
                left->count_offsets[left->count] * sizeof(uint16_t)) == 0 &&
         memcmp(left->neighbors, right->neighbors,
                left->neighbor_offsets[left->count] * sizeof(uint32_t)) == 0;
}

int main(void) {
  HnswBuildArena single;
  HnswBuildArena parallel;
  HnswBuildArena limited;
  HnswBuildArena cancelled;
  const int thread_counts[] = {4, 8, 16};
  int rc = initialize(&single, 1);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "single-thread arena failed: %d\n", rc);
    return 1;
  }
  for (size_t index = 0;
       index < sizeof(thread_counts) / sizeof(thread_counts[0]); ++index) {
    rc = initialize(&parallel, thread_counts[index]);
    if (rc != SQLITE_OK || !arenas_equal(&single, &parallel)) {
      fprintf(stderr, "arena output differs with %d threads\n",
              thread_counts[index]);
      hnsw_build_arena_destroy(&single);
      hnsw_build_arena_destroy(&parallel);
      return 1;
    }
    hnsw_build_arena_destroy(&parallel);
  }
  for (uint32_t node = 0; node < single.count; ++node) {
    for (int layer = 0; layer <= single.levels[node]; ++layer) {
      int count = 0;
      const uint32_t *neighbors =
          hnsw_build_arena_neighbors(&single, node, layer, &count);
      int maximum = layer == 0 ? single.m * 2 : single.m;
      if (count < 0 || count > maximum) {
        fprintf(stderr, "invalid neighbor count\n");
        return 1;
      }
      for (int i = 0; i < count; ++i) {
        if (neighbors[i] >= single.count || neighbors[i] == node) {
          fprintf(stderr, "invalid neighbor id\n");
          return 1;
        }
      }
    }
  }
  rc = hnsw_build_arena_init(&limited, 1000000, 128, 16, 128, HNSW_METRIC_L2,
                             16, 64U * 1024U * 1024U);
  if (rc != SQLITE_TOOBIG || limited.required_bytes <= limited.memory_limit) {
    fprintf(stderr, "arena memory limit was not enforced\n");
    return 1;
  }
  hnsw_build_arena_destroy(&limited);
  rc = hnsw_build_arena_init(&cancelled, 1, 2, 8, 64, HNSW_METRIC_L2, 1,
                             128U * 1024U * 1024U);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "cancelled arena initialization failed\n");
    return 1;
  }
  cancelled.node_ids[0] = 1;
  cancelled.levels[0] = 0;
  cancelled.vectors[0] = 0.0F;
  cancelled.vectors[1] = 1.0F;
  cancelled.norms[0] = 1.0;
  cancelled.is_cancelled = always_cancelled;
  if (hnsw_build_arena_prepare(&cancelled) != SQLITE_OK ||
      hnsw_build_arena_run(&cancelled) != SQLITE_INTERRUPT) {
    fprintf(stderr, "arena cancellation was not honored\n");
    return 1;
  }
  hnsw_build_arena_destroy(&cancelled);
  hnsw_build_arena_destroy(&single);
  puts("build arena tests passed");
  return 0;
}
