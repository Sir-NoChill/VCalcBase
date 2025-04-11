# Gather our source files in this directory.
set(
  vcalc_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/BackEnd.cpp"
)

# Build our executable from the source files.
add_executable(vcalc ${vcalc_src_files})
target_include_directories(vcalc PUBLIC ${ANTLR_GEN_DIR})

# Ensure that the antlr4-runtime is available.
add_dependencies(vcalc antlr)

# Find the libraries that correspond to the LLVM components
# that we wish to use
# NOTE that this is a bit of a hack. Generally you should
#  link using the find_project(...) and then add your
#  required libraries to the target_link_libraries
#  call, but this way it will just dynamically adjust
#  the linkage to all built targets (so that
#  things like llvm::InitializeAllTargets()) works
execute_process(COMMAND llvm-config --libfiles
  OUTPUT_VARIABLE llvm_libs
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE " " ";" llvm_libs ${llvm_libs})
message(VERBOSE "Found LLVM libraries: ${llvm_libs}")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# Add the MLIR, LLVM, antlr runtime and parser as libraries to link.
target_link_libraries(vcalc PRIVATE
    parser
    antlr4-runtime
    ${llvm_libs}
    ${dialect_libs}
    ${conversion_libs}
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    # MLIRAnalysis
    # MLIRCallInterfaces
    # MLIRCastInterfaces
    # MLIRExecutionEngine
    # MLIRFunctionInterfaces
    # MLIRIR
    # MLIRMemRefDialect
    # MLIRParser
    # MLIRPass
    # MLIRSideEffectInterfaces
    # MLIRTargetLLVMIRExport
    # MLIRTransforms    
)

# Symbolic link our executable to the base directory so we don't have to go searching for it.
symlink_to_bin("vcalc")
