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
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/IR/Verifier.h"

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

// Lowering to binary
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"

// LLVM Libs
#include "llvm/ADT/StringRef.h"

// standard
#include <cassert>
#include <iostream>
#include <memory>
#include <optional>

class BackEnd {
 public:
    BackEnd();

    int emitModule();
    int lowerDialects();
    int emitLLVM();
    int emitBinary(const char* filename);
 
 private:
    // MLIR
    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    mlir::ModuleOp module;
    std::shared_ptr<mlir::OpBuilder> builder;
    mlir::Location loc;
};
