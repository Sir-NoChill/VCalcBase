//===- VCalcTypes.td - Standalone dialect -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file is further licensed as teaching material for the University of
// Alberta under MIT
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_VCALC_DIALECT_TYPES
#define MLIR_VCALC_DIALECT_TYPES

include "Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

//===----------------------------------------------------------------------===//
// VCalc Types
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VCalc Generic Type
//===----------------------------------------------------------------------===//

class VCalc_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<VCalc_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// VCalc Integer
//===----------------------------------------------------------------------===//

// Strictly speaking, we don't really need this type. We could simply use the
//  builtin i32 type, but for the sake of completeness I will include a wrapper
//  so that we can extend it if we ever decide to do so.
def Int_Type : VCalc_Type<"Integer", "int"> {
  let summary = "VCalc integer type";
  let description = [{
    The vcalc integer type, this is the only type that can be explicitly
    instantiated by a constant op in any user program. There are no other
    types exposed to the user.
  }];
}

#endif // MLIR_VCALC_DIALECT_TYPES
