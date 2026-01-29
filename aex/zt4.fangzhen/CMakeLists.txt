
#
# simple-backend

set(TEST_TARGET zt4)
add_executable(${TEST_TARGET} simple-sgemm.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_compile_definitions(GGML_BACKEND_DL)

if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_ZLPU)
    add_compile_definitions(GGML_USE_ZLPU)
endif()
