add_executable(test_buffers test_buffers.c)
target_link_libraries(test_buffers tinygl_synth)

add_executable(test_determinism test_determinism.c)
target_link_libraries(test_determinism tinygl_synth)

if(UNIX)
    target_link_libraries(test_buffers m)
    target_link_libraries(test_determinism m)
endif()

if(WIN32)
    add_custom_command(TARGET test_buffers POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:tinygl_synth>
            $<TARGET_FILE_DIR:test_buffers>)
    add_custom_command(TARGET test_determinism POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:tinygl_synth>
            $<TARGET_FILE_DIR:test_determinism>)
endif()

add_test(NAME test_buffers COMMAND test_buffers)
add_test(NAME test_determinism COMMAND test_determinism)
