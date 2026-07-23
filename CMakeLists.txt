cmake_minimum_required(VERSION 3.18)
project(OjasX LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler optimization flags
if (MSVC)
    add_compile_options(/W4 /O2 /permissive- /std:c++20)
else()
    add_compile_options(-Wall -Wextra -O3 -std=c++20)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

# Find optional Vulkan and OpenCL packages
find_package(Vulkan QUIET)
find_package(OpenCL QUIET)

if (Vulkan_FOUND)
    message(STATUS "⚡ OjasX: Vulkan SDK found - Enabling SPIR-V Backend Functor")
    add_definitions(-DOJASX_HAS_VULKAN)
else()
    message(STATUS "⚠️ OjasX: Vulkan SDK not found - SPIR-V Backend operating in Emulated/JIT Mode")
endif()

if (OpenCL_FOUND)
    message(STATUS "⚡ OjasX: OpenCL SDK found - Enabling OpenCL 3.0 Backend Functor")
    add_definitions(-DOJASX_HAS_OPENCL)
else()
    message(STATUS "⚠️ OjasX: OpenCL SDK not found - OpenCL Backend operating in Emulated Mode")
endif()

# Core C++ Category Engine Test Suite
add_executable(test_category_core
    tests/test_category_core.cpp
)

if (Vulkan_FOUND)
    target_link_libraries(test_category_core PRIVATE Vulkan::Vulkan)
endif()

if (OpenCL_FOUND)
    target_link_libraries(test_category_core PRIVATE OpenCL::OpenCL)
endif()
