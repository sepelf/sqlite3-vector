cmake_minimum_required(VERSION 3.20)

project(sqlite_hnsw VERSION 0.1.0 LANGUAGES C)

configure_file(
  src/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/sqlite_hnsw_version.h
  @ONLY
)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_package(SQLite3 3.41 REQUIRED)
find_package(Threads REQUIRED)

add_library(sqlite_hnsw MODULE
  src/build_arena.c
  src/integrity.c
  src/sqlite_hnsw.c
  src/vector.c
)

target_include_directories(sqlite_hnsw PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${SQLite3_INCLUDE_DIRS}
)

target_link_libraries(sqlite_hnsw PRIVATE m Threads::Threads)

set_target_properties(sqlite_hnsw PROPERTIES
  PREFIX ""
  OUTPUT_NAME "sqlite_hnsw"
)

install(TARGETS sqlite_hnsw LIBRARY DESTINATION lib)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(sqlite_hnsw PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wstrict-prototypes -Wmissing-prototypes
  )
endif()
include(CTest)
if(BUILD_TESTING)
  add_executable(sqlite_hnsw_test tests/test_extension.c)
  target_include_directories(sqlite_hnsw_test PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_link_libraries(sqlite_hnsw_test PRIVATE SQLite::SQLite3 m)
  add_executable(sqlite_hnsw_integrity_test tests/test_integrity.c)
  target_link_libraries(sqlite_hnsw_integrity_test PRIVATE SQLite::SQLite3 m)
  add_executable(sqlite_hnsw_fault_vfs_test tests/test_fault_vfs.c)
  target_link_libraries(sqlite_hnsw_fault_vfs_test PRIVATE SQLite::SQLite3 m)
  add_executable(sqlite_hnsw_correctness_test tests/test_correctness.c)
  target_link_libraries(sqlite_hnsw_correctness_test PRIVATE SQLite::SQLite3 m)
  add_executable(sqlite_hnsw_state_model_test tests/test_state_model.c)
  target_link_libraries(sqlite_hnsw_state_model_test PRIVATE SQLite::SQLite3 m)
  add_executable(sqlite_hnsw_arena_test
    tests/test_build_arena.c
    src/build_arena.c
  )
  target_include_directories(sqlite_hnsw_arena_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${SQLite3_INCLUDE_DIRS}
  )
  target_link_libraries(sqlite_hnsw_arena_test PRIVATE Threads::Threads m)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sqlite_hnsw_test PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(sqlite_hnsw_integrity_test PRIVATE
      -Wall -Wextra -Wpedantic
    )
    target_compile_options(sqlite_hnsw_fault_vfs_test PRIVATE
      -Wall -Wextra -Wpedantic
    )
    target_compile_options(sqlite_hnsw_correctness_test PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
    target_compile_options(sqlite_hnsw_state_model_test PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
    target_compile_options(sqlite_hnsw_arena_test PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
  endif()
  add_test(
    NAME extension
    COMMAND sqlite_hnsw_test $<TARGET_FILE:sqlite_hnsw>
  )
  add_test(
    NAME integrity
    COMMAND sqlite_hnsw_integrity_test $<TARGET_FILE:sqlite_hnsw>
  )
  add_test(
    NAME fault_vfs
    COMMAND sqlite_hnsw_fault_vfs_test $<TARGET_FILE:sqlite_hnsw>
  )
  add_test(
    NAME correctness
    COMMAND sqlite_hnsw_correctness_test $<TARGET_FILE:sqlite_hnsw>
  )
  add_test(
    NAME state_model
    COMMAND sqlite_hnsw_state_model_test $<TARGET_FILE:sqlite_hnsw>
  )
  add_test(NAME build_arena COMMAND sqlite_hnsw_arena_test)
endif()

