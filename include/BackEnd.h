#pragma once

// Pass manager
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

// Translation
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/raw_os_ostream.h"

// MLIR IR
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

// Dialects 
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "VCalc/Dialect.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

// Passes
#include "VCalc/Passes.h"

// standard
#include <cassert>

class BackEnd {
 public:
    BackEnd();

    int emitModule();
    int lowerDialects();
    void dumpLLVM(std::ostream &os);
 
 private:
    // MLIR
    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    mlir::ModuleOp module;
    std::shared_ptr<mlir::OpBuilder> builder;
    mlir::Location loc;
};
