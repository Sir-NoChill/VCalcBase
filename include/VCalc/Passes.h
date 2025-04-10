//===- Passes.h - VCalc Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for VCalc.
//
//===----------------------------------------------------------------------===//

#ifndef VCALC_PASSES_H
#define VCALC_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace vcalc {
/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the VCalc IR (e.g. vectorAdd).
// std::unique_ptr<mlir::Pass> createLowerToAffinePass();

/// Create a pass for lowering operations the remaining `VCalc` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // namespace vcalc
} // namespace mlir

#endif // VCALC_PASSES_H
