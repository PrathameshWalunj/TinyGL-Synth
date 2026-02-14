add_executable(basic_cube basic_cube.c)
target_link_libraries(basic_cube tinygl_synth)

if(UNIX)
    target_link_libraries(basic_cube m)
endif()

add_executable(benchmark benchmark.c)
target_link_libraries(benchmark tinygl_synth)

if(UNIX)
    target_link_libraries(benchmark m)
endif()

if(WIN32)
    add_custom_command(TARGET basic_cube POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:tinygl_synth>
            $<TARGET_FILE_DIR:basic_cube>)
    add_custom_command(TARGET benchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:tinygl_synth>
            $<TARGET_FILE_DIR:benchmark>)
endif()
