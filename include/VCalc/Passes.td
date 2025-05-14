//===- VCalcPasses.td - Standalone dialect -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file is further licensed as teaching material for the University of
// Alberta under MIT
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_VCALC_DIALECT_PASSES
#define MLIR_VCALC_DIALECT_PASSES

include "mlir/Pass/PassBase.td"

def VCalcToMLIR : Pass<"vcalc-to-standard"> {
  let summary = "Lower `vcalc` to LLVM Dialect MLIR";
  let description = [{
    Materialize the vcalc types and operations into LLVM dialect operations
    and types. After this pass, there should be no operations remaining for
    conversion.
  }];

  let dependentDialects = [
    // TODO add any additional dialects that your dialect depends on here
    "mlir::LLVM::LLVMDialect",
  ];
}

#endif // MLIR_VCALC_DIALECT_PASSES

