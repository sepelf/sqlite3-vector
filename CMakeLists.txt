cmake_minimum_required(VERSION 3.16)

project(sqlite3_vector_hnsw LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_package(SQLite3 REQUIRED)
find_package(Threads REQUIRED)
include(CheckCCompilerFlag)
include(CTest)

check_c_compiler_flag("-mavx2" VECHNSW_HAS_AVX2_FLAG)
check_c_compiler_flag("-mfma" VECHNSW_HAS_FMA_FLAG)

add_library(vechnsw MODULE src/vechnsw.c)
target_include_directories(vechnsw PRIVATE ${SQLite3_INCLUDE_DIRS})
target_link_libraries(vechnsw PRIVATE m Threads::Threads)
target_compile_options(vechnsw PRIVATE
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-ffast-math>
  $<$<BOOL:${VECHNSW_HAS_AVX2_FLAG}>:-mavx2>
  $<$<BOOL:${VECHNSW_HAS_FMA_FLAG}>:-mfma>
)
set_target_properties(vechnsw PROPERTIES
  PREFIX ""
  OUTPUT_NAME "vechnsw"
)

add_executable(test_vechnsw tests/test_vechnsw.c)
target_include_directories(test_vechnsw PRIVATE ${SQLite3_INCLUDE_DIRS})
target_link_libraries(test_vechnsw PRIVATE SQLite::SQLite3 m)
target_compile_definitions(test_vechnsw PRIVATE TEST_EXTENSION_PATH="$<TARGET_FILE:vechnsw>")

add_executable(bench_vechnsw bench/bench_vechnsw.c)
target_include_directories(bench_vechnsw PRIVATE ${SQLite3_INCLUDE_DIRS})
target_link_libraries(bench_vechnsw PRIVATE SQLite::SQLite3 m)
target_compile_options(bench_vechnsw PRIVATE
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-ffast-math>
  $<$<BOOL:${VECHNSW_HAS_AVX2_FLAG}>:-mavx2>
  $<$<BOOL:${VECHNSW_HAS_FMA_FLAG}>:-mfma>
)
target_compile_definitions(bench_vechnsw PRIVATE TEST_EXTENSION_PATH="$<TARGET_FILE:vechnsw>")

add_executable(bench_sift bench/bench_sift.c)
target_include_directories(bench_sift PRIVATE ${SQLite3_INCLUDE_DIRS})
target_link_libraries(bench_sift PRIVATE SQLite::SQLite3 m)
target_compile_options(bench_sift PRIVATE
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-ffast-math>
  $<$<BOOL:${VECHNSW_HAS_AVX2_FLAG}>:-mavx2>
  $<$<BOOL:${VECHNSW_HAS_FMA_FLAG}>:-mfma>
)
target_compile_definitions(bench_sift PRIVATE TEST_EXTENSION_PATH="$<TARGET_FILE:vechnsw>")

if(BUILD_TESTING)
  add_test(NAME vechnsw_all COMMAND test_vechnsw)
  add_test(NAME vechnsw_scalar COMMAND test_vechnsw scalar)
  add_test(NAME vechnsw_virtual COMMAND test_vechnsw virtual)
  add_test(NAME vechnsw_cache COMMAND test_vechnsw cache)
  add_test(NAME vechnsw_schema COMMAND test_vechnsw schema)
  add_test(NAME vechnsw_shadow COMMAND test_vechnsw shadow)
  add_test(NAME vechnsw_txn COMMAND test_vechnsw txn)
  add_test(NAME vechnsw_planner COMMAND test_vechnsw planner)
endif()
