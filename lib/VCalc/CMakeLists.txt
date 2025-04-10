add_mlir_dialect_library(MLIRVCalc
  Ops.cpp
  LowerToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/VCalc

  DEPENDS
  VCalcOpsIncGen

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
