//===- VCalcDialect.h - Standalone dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file is further licensed as teaching material for the University of
// Alberta under MIT
//
//===----------------------------------------------------------------------===//

#ifndef VCALC_STANDALONEDIALECT_H
#define VCALC_STANDALONEDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the vcalc
/// dialect.
#include "VCalc/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// vcalc operations.
#define GET_OP_CLASSES
#include "VCalc/Ops.h.inc"

#endif // VCALC_STANDALONEDIALECT_H
