cmake_minimum_required(VERSION 3.27)
project(bezier_curve)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_library(bezier_curve SHARED
        bezier_curve/bezier_curve.cpp
        bezier_curve/bezier_curve_binding.cpp
)


if(APPLE)
    message(STATUS "Compiling on macOS")
    target_compile_definitions(bezier_curve PRIVATE MAC_OS)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        message(STATUS "Compiling on ARM64 architecture")
        target_compile_definitions(bezier_curve PRIVATE MAC_OS_ARM64)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        message(STATUS "Compiling on x86_64 architecture")
        target_compile_definitions(bezier_curve PRIVATE MAC_OS_X86_64)
    endif()
endif()


if(CMAKE_COMPILE_TARGET_LIB)
    install(TARGETS bezier_curve DESTINATION ${CMAKE_COMPILE_TARGET_LIB})
else()
    install(TARGETS bezier_curve)
endif()
