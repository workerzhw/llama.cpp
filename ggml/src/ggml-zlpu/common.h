/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef ZLPU_COMMON_H
#define ZLPU_COMMON_H


#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <functional>
#include <optional>
#include <list>

#include "../include/ggml-zlpu.h"
#include "../include/ggml.h"
#include "../ggml-impl.h"

#define MATRIX_ROW_PADDING    512
#define GGML_ZLPU_MAX_STREAMS 8

/**
 * @brief Handles ZLPU-related errors by printing an error message and
 *        terminating the program.
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number at which the error occurred.
 * @param msg The error message.
 */
[[noreturn]] void ggml_zlpu_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

/**
 * @brief Checks the result of a ZLPU function call and invokes the error
 *        handler if the call fails.
 * @param stmt The ZLPU function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#define ACL_CHECK_GEN(stmt, success, error_fn)                                \
    do {                                                                      \
        int err_code = (stmt);                                                \
        if (err_code != (success)) {                                          \
            ggml_zlpu_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } while (0);

#define ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, zakGetRecentErrMsg)

/**
 * @brief Contains information about ZLPU devices.
 */
struct ggml_zlpu_device_info {
    /**
     * @brief Number of ZLPU devices available.
     */
    int32_t device_count;

};

const ggml_zlpu_device_info & ggml_zlpu_info();

void    ggml_zlpu_set_device(int32_t device);
struct ggml_backend_zlpu_context {
    int32_t     device;               /**< Device ID. */
    std::string name;                 /**< Name of the device. */
    std::string description;          /**< Description of the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_zlpu_context(int device) :
        device(device),
        name("ZLPU" + std::to_string(device)) {
        ggml_zlpu_set_device(device);
        description = "wmv701";

    }

    ~ggml_backend_zlpu_context() {
    }
};

#endif  // ZLPU_COMMON_H
