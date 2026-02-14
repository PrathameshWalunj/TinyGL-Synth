cmake_minimum_required(VERSION 3.15)

project(tinygl-synth
    VERSION 0.1.0
    DESCRIPTION "CPU-based synthetic data generator for embodied AI"
    LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(TINYGL_SYNTH_BUILD_EXAMPLES "Build example programs" ON)
option(TINYGL_SYNTH_BUILD_TESTS "Build unit tests" ON)
option(TINYGL_SYNTH_BUILD_PYTHON "Build Python bindings" ON)

if(MSVC)
    add_compile_options(/W4)
    add_compile_options(/wd4204)
else()
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-function
        -ffast-math
    )
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g3 -DDEBUG")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

set(TINYGL_SYNTH_SOURCES
    src/core/synth_context.c
    src/core/synth_render.c
    src/raster/zbuffer.c
    src/raster/triangle.c
    src/raster/triangle_synth.c
    src/math/matrix.c
    src/math/vector.c
    src/api/synth_api.c
)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)


add_library(tinygl_synth SHARED ${TINYGL_SYNTH_SOURCES})

if(UNIX)
    target_link_libraries(tinygl_synth PRIVATE m)
endif()

if(TINYGL_SYNTH_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(TINYGL_SYNTH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(TARGETS tinygl_synth
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

install(DIRECTORY include/tinygl_synth
    DESTINATION include
    FILES_MATCHING PATTERN "*.h")

message(STATUS "=== tinygl-synth Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build examples: ${TINYGL_SYNTH_BUILD_EXAMPLES}")
message(STATUS "Build tests: ${TINYGL_SYNTH_BUILD_TESTS}")
message(STATUS "Build Python bindings: ${TINYGL_SYNTH_BUILD_PYTHON}")
message(STATUS "===================================")