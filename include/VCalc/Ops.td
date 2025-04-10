//===- VCalcDialect.td - Toy dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VCALC_DIALECT
#define VCALC_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

//===----------------------------------------------------------------------===//
// VCalc dialect definition.
//===----------------------------------------------------------------------===//

def VCalc_Dialect : Dialect {
    let name = "vcalc";
    let summary = "A vcalc out-of-tree MLIR dialect.";
    let description = [{
      A dialect for the VCalc language
    }];
    let cppNamespace = "::mlir::vcalc";
}

//===----------------------------------------------------------------------===//
// VCalc Op definitions
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Base vcalc operation definition.
//===----------------------------------------------------------------------===//

class VCalc_Op<string mnemonic, list<Trait> traits = []> :
        Op<VCalc_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : VCalc_Op<"print", [Pure]> {
  let summary = "Print op";
  let description = [{
    The `print` op prints a vcalc object.

    ```mlir
    %0 = vcalc.constant ...
    vcalc.print(%0 : tensor<1x2xf64>)
    ```
  }];

  let assemblyFormat = "attr-dict `(``)` `->` `(``)`";
}

#endif // VCALC_DIALECT
