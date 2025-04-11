# Need MLIR so we can link against it. Either we automatically find this in the "default" place or
# it's found because we set up $MLIR_DIR. There should be no additions necessary here.
find_package(MLIR REQUIRED CONFIG)

# Status messages about LLVM found.
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

set(LLVM_BINARY_DIR ${LLVM_DIR}/../../../bin)

# Add the mlir utils
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include(HandleLLVMOptions)

# Add more LLVM includes
set(VCALC_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(VCALC_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${VCALC_BINARY_DIR}/include)
include_directories(${VCALC_SOURCE_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
