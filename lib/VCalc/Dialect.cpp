//===- VCalcDialect.cpp - Standalone dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file is further licensed as teaching material for the University of
// Alberta under MIT
//
//===----------------------------------------------------------------------===//
//
#include "VCalc/Dialect.h"

//===----------------------------------------------------------------------===//
// Toy dialect
//===----------------------------------------------------------------------===//

void mlir::vcalc::VCalcDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "VCalc/Ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "VCalc/Types.cpp.inc"
      >();
}
