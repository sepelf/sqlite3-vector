#include "build_arena.h"

#include "vector.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARENA_NO_NODE UINT32_MAX
#define ARENA_SEED_COUNT 1024U
#define ARENA_BATCH_SIZE 64U
#define ARENA_MAX_LEVEL 32

typedef struct ArenaItem {
  uint32_t node;
  float distance;
} ArenaItem;

typedef struct ArenaHeap {
  ArenaItem *items;
  int count;
  int capacity;
  int is_min;
} ArenaHeap;

typedef struct ArenaWorkspace {
  uint32_t *visited;
  uint32_t epoch;
  ArenaHeap candidates;
  ArenaHeap nearest;
  ArenaItem *results;
  int result_capacity;
  ArenaItem *scratch;
  int scratch_capacity;
} ArenaWorkspace;

typedef struct Backlink {
  uint32_t owner;
  uint32_t neighbor;
  uint16_t layer;
} Backlink;

typedef struct BacklinkGroup {
  int start;
  int count;
} BacklinkGroup;

typedef struct WorkerJob {
  HnswBuildArena *arena;
  ArenaWorkspace *workspaces;
  int worker_index;
  int worker_count;
  int phase;
  uint32_t batch_start;
  uint32_t batch_count;
  uint32_t snapshot_count;
  uint32_t snapshot_entry;
  int snapshot_max_level;
  Backlink *backlinks;
  BacklinkGroup *groups;
  int group_count;
  int rc;
} WorkerJob;

static int checked_add(size_t *value, size_t addition) {
  if (*value > SIZE_MAX - addition) {
    return SQLITE_TOOBIG;
  }
  *value += addition;
  return SQLITE_OK;
}

static int checked_array(size_t *value, size_t count, size_t width) {
  if (width != 0 && count > SIZE_MAX / width) {
    return SQLITE_TOOBIG;
  }
  return checked_add(value, count * width);
}

static size_t count_offset(const HnswBuildArena *arena, uint32_t node,
                           int layer) {
  return arena->count_offsets[node] + (size_t)layer;
}

static size_t neighbor_offset(const HnswBuildArena *arena, uint32_t node,
                              int layer) {
  size_t offset = arena->neighbor_offsets[node];
  return layer == 0 ? offset
                    : offset + (size_t)(2 * arena->m) +
                          (size_t)(layer - 1) * (size_t)arena->m;
}

static int layer_capacity(const HnswBuildArena *arena, int layer) {
  return layer == 0 ? arena->m * 2 : arena->m;
}

static float arena_distance(const HnswBuildArena *arena, const float *query,
                            double query_norm, uint32_t node) {
  const float *candidate = arena->vectors + (size_t)node * (size_t)arena->dims;
  return hnsw_vector_distance_fast(query, query_norm, candidate,
                                   arena->norms[node], arena->dims,
                                   arena->metric);
}

static float nodes_distance(const HnswBuildArena *arena, uint32_t left,
                            uint32_t right) {
  const float *a = arena->vectors + (size_t)left * (size_t)arena->dims;
  const float *b = arena->vectors + (size_t)right * (size_t)arena->dims;
  return hnsw_vector_distance_fast(a, arena->norms[left], b,
                                   arena->norms[right], arena->dims,
                                   arena->metric);
}

static int item_better(const ArenaHeap *heap, ArenaItem a, ArenaItem b) {
  if (a.distance == b.distance) {
    return heap->is_min ? a.node < b.node : a.node > b.node;
  }
  return heap->is_min ? a.distance < b.distance : a.distance > b.distance;
}

static int heap_reserve(ArenaHeap *heap, int capacity) {
  ArenaItem *grown = NULL;
  int next = heap->capacity == 0 ? 256 : heap->capacity;
  while (next < capacity) {
    if (next > INT32_MAX / 2) {
      return SQLITE_TOOBIG;
    }
    next *= 2;
  }
  if (next == heap->capacity) {
    return SQLITE_OK;
  }
  grown = realloc(heap->items, (size_t)next * sizeof(*grown));
  if (grown == NULL) {
    return SQLITE_NOMEM;
  }
  heap->items = grown;
  heap->capacity = next;
  return SQLITE_OK;
}

static int heap_push(ArenaHeap *heap, ArenaItem item) {
  int index = 0;
  int rc = heap_reserve(heap, heap->count + 1);
  if (rc != SQLITE_OK) {
    return rc;
  }
  index = heap->count++;
  heap->items[index] = item;
  while (index > 0) {
    int parent = (index - 1) / 2;
    ArenaItem swap;
    if (!item_better(heap, heap->items[index], heap->items[parent])) {
      break;
    }
    swap = heap->items[index];
    heap->items[index] = heap->items[parent];
    heap->items[parent] = swap;
    index = parent;
  }
  return SQLITE_OK;
}

static ArenaItem heap_pop(ArenaHeap *heap) {
  ArenaItem result = heap->items[0];
  int index = 0;
  --heap->count;
  if (heap->count == 0) {
    return result;
  }
  heap->items[0] = heap->items[heap->count];
  for (;;) {
    int left = index * 2 + 1;
    int right = left + 1;
    int best = index;
    ArenaItem swap;
    if (left < heap->count &&
        item_better(heap, heap->items[left], heap->items[best])) {
      best = left;
    }
    if (right < heap->count &&
        item_better(heap, heap->items[right], heap->items[best])) {
      best = right;
    }
    if (best == index) {
      break;
    }
    swap = heap->items[index];
    heap->items[index] = heap->items[best];
    heap->items[best] = swap;
    index = best;
  }
  return result;
}

static int compare_items(const void *left, const void *right) {
  const ArenaItem *a = left;
  const ArenaItem *b = right;
  if (a->distance < b->distance) {
    return -1;
  }
  if (a->distance > b->distance) {
    return 1;
  }
  return a->node < b->node ? -1 : (a->node > b->node ? 1 : 0);
}

static int ensure_items(ArenaItem **items, int *capacity, int required) {
  ArenaItem *grown = NULL;
  int next = *capacity == 0 ? 256 : *capacity;
  while (next < required) {
    if (next > INT32_MAX / 2) {
      return SQLITE_TOOBIG;
    }
    next *= 2;
  }
  if (next == *capacity) {
    return SQLITE_OK;
  }
  grown = realloc(*items, (size_t)next * sizeof(*grown));
  if (grown == NULL) {
    return SQLITE_NOMEM;
  }
  *items = grown;
  *capacity = next;
  return SQLITE_OK;
}

static const uint32_t *neighbors_at(const HnswBuildArena *arena, uint32_t node,
                                    int layer, int *count) {
  if (node >= arena->count || layer < 0 || layer > arena->levels[node]) {
    *count = 0;
    return NULL;
  }
  *count = arena->neighbor_counts[count_offset(arena, node, layer)];
  return arena->neighbors + neighbor_offset(arena, node, layer);
}

static uint32_t *mutable_neighbors(HnswBuildArena *arena, uint32_t node,
                                   int layer, uint16_t **count) {
  *count = &arena->neighbor_counts[count_offset(arena, node, layer)];
  return arena->neighbors + neighbor_offset(arena, node, layer);
}

static int next_epoch(ArenaWorkspace *workspace, size_t count) {
  ++workspace->epoch;
  if (workspace->epoch == 0) {
    memset(workspace->visited, 0, count * sizeof(*workspace->visited));
    workspace->epoch = 1;
  }
  return SQLITE_OK;
}

static int search_layer(const HnswBuildArena *arena, const float *query,
                        double query_norm, uint32_t entry, int layer, int ef,
                        uint32_t linked_count, ArenaWorkspace *workspace,
                        ArenaItem **results, int *result_count) {
  ArenaHeap *candidates = &workspace->candidates;
  ArenaHeap *nearest = &workspace->nearest;
  ArenaItem initial;
  int rc = SQLITE_OK;
  candidates->count = 0;
  candidates->is_min = 1;
  nearest->count = 0;
  nearest->is_min = 0;
  next_epoch(workspace, arena->count);
  initial.node = entry;
  initial.distance = arena_distance(arena, query, query_norm, entry);
  workspace->visited[entry] = workspace->epoch;
  if ((rc = heap_push(candidates, initial)) != SQLITE_OK ||
      (rc = heap_push(nearest, initial)) != SQLITE_OK) {
    return rc;
  }
  while (candidates->count > 0) {
    ArenaItem current = heap_pop(candidates);
    const uint32_t *edges = NULL;
    int edge_count = 0;
    int i = 0;
    if (nearest->count >= ef && current.distance > nearest->items[0].distance) {
      break;
    }
    edges = neighbors_at(arena, current.node, layer, &edge_count);
    for (i = 0; i < edge_count; ++i) {
      ArenaItem candidate;
      uint32_t node = edges[i];
      if (node >= linked_count ||
          workspace->visited[node] == workspace->epoch) {
        continue;
      }
      workspace->visited[node] = workspace->epoch;
      candidate.node = node;
      candidate.distance = arena_distance(arena, query, query_norm, node);
      if (nearest->count < ef ||
          candidate.distance <= nearest->items[0].distance) {
        if ((rc = heap_push(candidates, candidate)) != SQLITE_OK ||
            (rc = heap_push(nearest, candidate)) != SQLITE_OK) {
          return rc;
        }
        if (nearest->count > ef) {
          (void)heap_pop(nearest);
        }
      }
    }
  }
  rc = ensure_items(&workspace->results, &workspace->result_capacity,
                    nearest->count);
  if (rc != SQLITE_OK) {
    return rc;
  }
  *result_count = nearest->count;
  for (int i = 0; i < *result_count; ++i) {
    workspace->results[i] = heap_pop(nearest);
  }
  qsort(workspace->results, (size_t)*result_count, sizeof(ArenaItem),
        compare_items);
  *results = workspace->results;
  return SQLITE_OK;
}

static uint32_t greedy(const HnswBuildArena *arena, const float *query,
                       double query_norm, uint32_t start, int layer) {
  uint32_t best = start;
  float best_distance = arena_distance(arena, query, query_norm, best);
  int changed = 1;
  while (changed) {
    const uint32_t *edges = NULL;
    int count = 0;
    int i = 0;
    changed = 0;
    edges = neighbors_at(arena, best, layer, &count);
    for (i = 0; i < count; ++i) {
      float distance = arena_distance(arena, query, query_norm, edges[i]);
      if (distance < best_distance ||
          (distance == best_distance && edges[i] < best)) {
        best = edges[i];
        best_distance = distance;
        changed = 1;
      }
    }
  }
  return best;
}

static int select_neighbors(const HnswBuildArena *arena, ArenaItem *candidates,
                            int count, int maximum, uint32_t *output,
                            uint16_t *output_count) {
  int selected = 0;
  int i = 0;
  qsort(candidates, (size_t)count, sizeof(*candidates), compare_items);
  for (i = 0; i < count && selected < maximum; ++i) {
    int diverse = 1;
    int j = 0;
    for (j = 0; j < selected; ++j) {
      if (nodes_distance(arena, candidates[i].node, output[j]) <
          candidates[i].distance) {
        diverse = 0;
        break;
      }
    }
    if (diverse) {
      output[selected++] = candidates[i].node;
    }
  }
  for (i = 0; i < count && selected < maximum; ++i) {
    int present = 0;
    int j = 0;
    for (j = 0; j < selected; ++j) {
      present |= output[j] == candidates[i].node;
    }
    if (!present) {
      output[selected++] = candidates[i].node;
    }
  }
  *output_count = (uint16_t)selected;
  return SQLITE_OK;
}

static int propose_node(HnswBuildArena *arena, uint32_t node,
                        uint32_t linked_count, uint32_t entry, int max_level,
                        ArenaWorkspace *workspace) {
  const float *query = arena->vectors + (size_t)node * (size_t)arena->dims;
  double norm = arena->norms[node];
  uint32_t current = entry;
  int level = arena->levels[node];
  int layer = 0;
  int rc = SQLITE_OK;
  if (linked_count == 0 || entry == ARENA_NO_NODE) {
    return SQLITE_OK;
  }
  for (layer = max_level; layer > level; --layer) {
    current = greedy(arena, query, norm, current, layer);
  }
  for (layer = level < max_level ? level : max_level; layer >= 0; --layer) {
    ArenaItem *results = NULL;
    int result_count = 0;
    uint16_t *selected_count = NULL;
    uint32_t *selected = mutable_neighbors(arena, node, layer, &selected_count);
    rc =
        search_layer(arena, query, norm, current, layer, arena->ef_construction,
                     linked_count, workspace, &results, &result_count);
    if (rc != SQLITE_OK) {
      return rc;
    }
    rc = select_neighbors(arena, results, result_count,
                          layer == 0 ? arena->m * 2 : arena->m, selected,
                          selected_count);
    if (rc != SQLITE_OK) {
      return rc;
    }
    if (result_count > 0) {
      current = results[0].node;
    }
  }
  return SQLITE_OK;
}

static int compare_backlinks(const void *left, const void *right) {
  const Backlink *a = left;
  const Backlink *b = right;
  if (a->owner != b->owner) {
    return a->owner < b->owner ? -1 : 1;
  }
  if (a->layer != b->layer) {
    return a->layer < b->layer ? -1 : 1;
  }
  return a->neighbor < b->neighbor ? -1 : (a->neighbor > b->neighbor ? 1 : 0);
}

static int apply_backlink_group(HnswBuildArena *arena, const Backlink *items,
                                int count, ArenaWorkspace *workspace) {
  uint16_t *stored_count = NULL;
  uint32_t *stored =
      mutable_neighbors(arena, items[0].owner, items[0].layer, &stored_count);
  int maximum = layer_capacity(arena, items[0].layer);
  int candidate_count = 0;
  int i = 0;
  int rc = ensure_items(&workspace->scratch, &workspace->scratch_capacity,
                        (int)*stored_count + count);
  if (rc != SQLITE_OK) {
    return rc;
  }
  for (i = 0; i < *stored_count; ++i) {
    workspace->scratch[candidate_count++].node = stored[i];
  }
  for (i = 0; i < count; ++i) {
    uint32_t node = items[i].neighbor;
    int j = 0;
    int present = 0;
    for (j = 0; j < candidate_count; ++j) {
      present |= workspace->scratch[j].node == node;
    }
    if (!present) {
      workspace->scratch[candidate_count++].node = node;
    }
  }
  for (i = 0; i < candidate_count; ++i) {
    workspace->scratch[i].distance =
        nodes_distance(arena, items[0].owner, workspace->scratch[i].node);
  }
  return select_neighbors(arena, workspace->scratch, candidate_count, maximum,
                          stored, stored_count);
}

static void *worker_main(void *argument) {
  WorkerJob *job = argument;
  ArenaWorkspace *workspace = &job->workspaces[job->worker_index];
  int i = 0;
  job->rc = SQLITE_OK;
  if (job->phase == 0) {
    for (i = job->worker_index; i < (int)job->batch_count;
         i += job->worker_count) {
      job->rc = propose_node(job->arena, job->batch_start + (uint32_t)i,
                             job->snapshot_count, job->snapshot_entry,
                             job->snapshot_max_level, workspace);
      if (job->rc != SQLITE_OK) {
        break;
      }
    }
  } else {
    for (i = job->worker_index; i < job->group_count; i += job->worker_count) {
      BacklinkGroup group = job->groups[i];
      job->rc = apply_backlink_group(job->arena, job->backlinks + group.start,
                                     group.count, workspace);
      if (job->rc != SQLITE_OK) {
        break;
      }
    }
  }
  return NULL;
}

static int run_workers(WorkerJob *jobs, int worker_count) {
  pthread_t threads[64];
  int created = 0;
  int i = 0;
  int rc = SQLITE_OK;
  for (i = 0; i < worker_count; ++i) {
    jobs[i].worker_index = i;
    jobs[i].worker_count = worker_count;
    if (pthread_create(&threads[i], NULL, worker_main, &jobs[i]) != 0) {
      rc = SQLITE_ERROR;
      break;
    }
    ++created;
  }
  for (i = 0; i < created; ++i) {
    (void)pthread_join(threads[i], NULL);
  }
  if (created != worker_count) {
    return rc;
  }
  for (i = 0; i < worker_count; ++i) {
    if (jobs[i].rc != SQLITE_OK) {
      return jobs[i].rc;
    }
  }
  return SQLITE_OK;
}

static int process_batch(HnswBuildArena *arena, uint32_t start,
                         uint32_t batch_count, ArenaWorkspace *workspaces) {
  Backlink *backlinks = NULL;
  BacklinkGroup *groups = NULL;
  WorkerJob jobs[64];
  uint32_t snapshot_entry = arena->entry;
  int snapshot_max = arena->max_level;
  int worker_count = arena->thread_count;
  int backlink_count = 0;
  int group_count = 0;
  int capacity = (int)batch_count * (2 * arena->m + ARENA_MAX_LEVEL * arena->m);
  uint32_t i = 0;
  int rc = SQLITE_OK;
  if (worker_count > (int)batch_count) {
    worker_count = (int)batch_count;
  }
  memset(jobs, 0, sizeof(jobs));
  for (int w = 0; w < worker_count; ++w) {
    jobs[w].arena = arena;
    jobs[w].workspaces = workspaces;
    jobs[w].worker_index = w;
    jobs[w].worker_count = worker_count;
    jobs[w].phase = 0;
    jobs[w].batch_start = start;
    jobs[w].batch_count = batch_count;
    jobs[w].snapshot_count = start;
    jobs[w].snapshot_entry = snapshot_entry;
    jobs[w].snapshot_max_level = snapshot_max;
  }
  rc = worker_count == 1 ? (worker_main(&jobs[0]), jobs[0].rc)
                         : run_workers(jobs, worker_count);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (capacity > 0) {
    backlinks = malloc((size_t)capacity * sizeof(*backlinks));
    if (backlinks == NULL) {
      return SQLITE_NOMEM;
    }
  }
  for (i = 0; i < batch_count; ++i) {
    uint32_t node = start + i;
    for (int layer = 0; layer <= arena->levels[node]; ++layer) {
      int count = 0;
      const uint32_t *neighbors = neighbors_at(arena, node, layer, &count);
      for (int j = 0; j < count; ++j) {
        backlinks[backlink_count++] =
            (Backlink){neighbors[j], node, (uint16_t)layer};
      }
    }
  }
  qsort(backlinks, (size_t)backlink_count, sizeof(*backlinks),
        compare_backlinks);
  if (backlink_count > 0) {
    groups = malloc((size_t)backlink_count * sizeof(*groups));
    if (groups == NULL) {
      free(backlinks);
      return SQLITE_NOMEM;
    }
    for (int offset = 0; offset < backlink_count;) {
      int end = offset + 1;
      while (end < backlink_count &&
             backlinks[end].owner == backlinks[offset].owner &&
             backlinks[end].layer == backlinks[offset].layer) {
        ++end;
      }
      groups[group_count++] = (BacklinkGroup){offset, end - offset};
      offset = end;
    }
    worker_count =
        arena->thread_count < group_count ? arena->thread_count : group_count;
    memset(jobs, 0, sizeof(jobs));
    for (int w = 0; w < worker_count; ++w) {
      jobs[w].arena = arena;
      jobs[w].workspaces = workspaces;
      jobs[w].worker_index = w;
      jobs[w].worker_count = worker_count;
      jobs[w].phase = 1;
      jobs[w].backlinks = backlinks;
      jobs[w].groups = groups;
      jobs[w].group_count = group_count;
    }
    rc = worker_count == 1 ? (worker_main(&jobs[0]), jobs[0].rc)
                           : run_workers(jobs, worker_count);
  }
  if (rc == SQLITE_OK) {
    for (i = 0; i < batch_count; ++i) {
      uint32_t node = start + i;
      if (arena->entry == ARENA_NO_NODE ||
          arena->levels[node] > arena->max_level) {
        arena->entry = node;
        arena->max_level = arena->levels[node];
      }
    }
  }
  free(backlinks);
  free(groups);
  return rc;
}

int hnsw_build_arena_init(HnswBuildArena *arena, size_t count, int dims, int m,
                          int ef_construction, int metric, int thread_count,
                          size_t memory_limit) {
  size_t required = sizeof(*arena);
  memset(arena, 0, sizeof(*arena));
  arena->count = count;
  arena->dims = dims;
  arena->m = m;
  arena->ef_construction = ef_construction;
  arena->metric = metric;
  arena->thread_count = thread_count;
  arena->memory_limit = memory_limit;
  arena->entry = ARENA_NO_NODE;
  arena->max_level = -1;
  if (checked_array(&required, count, sizeof(sqlite3_int64)) != SQLITE_OK ||
      checked_array(&required, count * (size_t)dims, sizeof(float)) !=
          SQLITE_OK ||
      checked_array(&required, count, sizeof(double)) != SQLITE_OK ||
      checked_array(&required, count, sizeof(unsigned char)) != SQLITE_OK ||
      checked_array(&required, count + 1U, sizeof(size_t)) != SQLITE_OK ||
      checked_array(&required, count + 1U, sizeof(size_t)) != SQLITE_OK ||
      checked_array(&required, count * (size_t)thread_count,
                    sizeof(uint32_t)) != SQLITE_OK) {
    return SQLITE_TOOBIG;
  }
  arena->required_bytes = required;
  if (required > memory_limit || count > UINT32_MAX) {
    return SQLITE_TOOBIG;
  }
  arena->node_ids = malloc(count * sizeof(*arena->node_ids));
  arena->vectors = malloc(count * (size_t)dims * sizeof(*arena->vectors));
  arena->norms = malloc(count * sizeof(*arena->norms));
  arena->levels = malloc(count * sizeof(*arena->levels));
  arena->neighbor_offsets = malloc((count + 1U) * sizeof(size_t));
  arena->count_offsets = malloc((count + 1U) * sizeof(size_t));
  if (arena->node_ids == NULL || arena->vectors == NULL ||
      arena->norms == NULL || arena->levels == NULL ||
      arena->neighbor_offsets == NULL || arena->count_offsets == NULL) {
    hnsw_build_arena_destroy(arena);
    return SQLITE_NOMEM;
  }
  return SQLITE_OK;
}

int hnsw_build_arena_prepare(HnswBuildArena *arena) {
  size_t neighbor_slots = 0;
  size_t count_slots = 0;
  size_t required = arena->required_bytes;
  for (size_t i = 0; i < arena->count; ++i) {
    size_t node_neighbors =
        (size_t)(2 * arena->m) + (size_t)arena->levels[i] * (size_t)arena->m;
    arena->neighbor_offsets[i] = neighbor_slots;
    arena->count_offsets[i] = count_slots;
    if (checked_add(&neighbor_slots, node_neighbors) != SQLITE_OK ||
        checked_add(&count_slots, (size_t)arena->levels[i] + 1U) != SQLITE_OK) {
      return SQLITE_TOOBIG;
    }
  }
  arena->neighbor_offsets[arena->count] = neighbor_slots;
  arena->count_offsets[arena->count] = count_slots;
  if (checked_array(&required, neighbor_slots, sizeof(uint32_t)) != SQLITE_OK ||
      checked_array(&required, count_slots, sizeof(uint16_t)) != SQLITE_OK) {
    return SQLITE_TOOBIG;
  }
  arena->required_bytes = required;
  if (required > arena->memory_limit) {
    return SQLITE_TOOBIG;
  }
  arena->neighbors = calloc(neighbor_slots, sizeof(*arena->neighbors));
  arena->neighbor_counts = calloc(count_slots, sizeof(*arena->neighbor_counts));
  if (arena->neighbors == NULL || arena->neighbor_counts == NULL) {
    return SQLITE_NOMEM;
  }
  return SQLITE_OK;
}

int hnsw_build_arena_run(HnswBuildArena *arena) {
  ArenaWorkspace *workspaces = NULL;
  uint32_t start = 0;
  int rc = SQLITE_OK;
  int trace = getenv("SQLITE_HNSW_TRACE_BUILD") != NULL;
  workspaces = calloc((size_t)arena->thread_count, sizeof(*workspaces));
  if (workspaces == NULL) {
    return SQLITE_NOMEM;
  }
  for (int i = 0; i < arena->thread_count; ++i) {
    workspaces[i].visited = calloc(arena->count, sizeof(uint32_t));
    if (workspaces[i].visited == NULL) {
      rc = SQLITE_NOMEM;
      goto done;
    }
  }
  while (start < arena->count && rc == SQLITE_OK) {
    uint32_t batch = start < ARENA_SEED_COUNT
                         ? 1U
                         : (uint32_t)(arena->count - start > ARENA_BATCH_SIZE
                                          ? ARENA_BATCH_SIZE
                                          : arena->count - start);
    rc = process_batch(arena, start, batch, workspaces);
    start += batch;
    if (rc == SQLITE_OK && arena->is_cancelled != NULL &&
        arena->is_cancelled(arena->cancel_context)) {
      rc = SQLITE_INTERRUPT;
    }
    if (trace && (start == arena->count || start % 256U == 0)) {
      fprintf(stderr, "sqlite-hnsw arena: %u/%zu nodes\n", start, arena->count);
    }
  }
done:
  for (int i = 0; i < arena->thread_count; ++i) {
    free(workspaces[i].visited);
    free(workspaces[i].candidates.items);
    free(workspaces[i].nearest.items);
    free(workspaces[i].results);
    free(workspaces[i].scratch);
  }
  free(workspaces);
  return rc;
}

const uint32_t *hnsw_build_arena_neighbors(const HnswBuildArena *arena,
                                           uint32_t node, int layer,
                                           int *count) {
  return neighbors_at(arena, node, layer, count);
}

void hnsw_build_arena_destroy(HnswBuildArena *arena) {
  free(arena->node_ids);
  free(arena->vectors);
  free(arena->norms);
  free(arena->levels);
  free(arena->neighbor_offsets);
  free(arena->count_offsets);
  free(arena->neighbors);
  free(arena->neighbor_counts);
  memset(arena, 0, sizeof(*arena));
}
