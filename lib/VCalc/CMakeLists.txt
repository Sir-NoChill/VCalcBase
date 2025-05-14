add_mlir_dialect_library(MLIRVCalc
  Dialect.cpp
  Ops.cpp
  Types.cpp
  LowerToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/VCalc

  DEPENDS
  VCalcOpsIncGen
  VCalcPassesIncGen

  # If you need more included MLIR things, add them here
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
  MLIRAnalysis
  MLIRFunctionInterfaces
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
)
