cmake_minimum_required(VERSION 3.23.1)
project(deft 
		VERSION 2024
		DESCRIPTION "An IO-aware fast attention kernel for efficient tree-structured interactions with LLMs"
		LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(Python3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
if(NOT Python3_FOUND)
  message(FATAL_ERROR "Python3 not found.")
endif()
if(NOT CUDAToolkit_FOUND)
  message(FATAL_ERROR "CUDA not found.")
endif()
message(STATUS "Python3 found at ${Python3_EXECUTABLE}")
message(STATUS "CUDA version is ${CUDAToolkit_VERSION}")
set(DEFT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
include_directories(${DEFT_SOURCE_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

set(DEFT_ENABLE_TESTS CACHE BOOL "Enable tests for DEFT" ON)

add_subdirectory(3rdparty/googletest)
if(DEFT_ENABLE_TESTS)
  enable_testing()
  set(DEFT_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/csrc)
  file(GLOB_RECURSE TEST_DUMMY_SRC ${DEFT_TEST_SOURCE_DIR}/test_dummy.cu)
  add_executable(test_dummy ${TEST_DUMMY_SRC})
  target_link_libraries(test_dummy gtest_main gtest)
  add_test(NAME test_dummy COMMAND test_dummy)
endif()

