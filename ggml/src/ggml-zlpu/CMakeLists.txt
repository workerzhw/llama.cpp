
set(SOC_VERSION "")

# Only Support Linux.
if (NOT UNIX)
    message(FATAL_ERROR "ZLPU: ZLPU toolkit supports unix but not ${CMAKE_SYSTEM_NAME}")
endif()

# Supported platforms: x86-64, arm64
if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
else()
    message(FATAL_ERROR "ZLPU: ZLPU toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Set header and libs
set(ZLPU_INCLUDE_DIRS
    ${ZLPU_INSTALL_DIR}/include
    # ${ZLPU_INSTALL_DIR}/include/aclnn
    # ${ZLPU_INSTALL_DIR}/acllib/include
)


file(GLOB GGML_SOURCES_ZLPU "*.cpp")

ggml_add_backend_library(ggml-zlpu ${GGML_SOURCES_ZLPU})
target_link_libraries(ggml-zlpu PRIVATE ${ZLPU_LIBRARIES})
target_include_directories(ggml-zlpu PRIVATE ${ZLPU_INCLUDE_DIRS})
target_link_directories(ggml-zlpu PRIVATE ${ZLPU_INSTALL_DIR}/lib64)

target_compile_definitions(ggml-zlpu PRIVATE "-D${SOC_TYPE_COMPILE_OPTION}")


message(STATUS "ZLPU: ZLPU_INCLUDE_DIRS =  ${ZLPU_INCLUDE_DIRS}")
# message(STATUS "ZLPU: ZLPU_LIBRARIES =  ${ZLPU_LIBRARIES}")
