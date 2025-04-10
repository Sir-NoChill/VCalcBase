#include "VCalc/Passes.h"
#include "VCalc/Dialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DerivedTypes.h"
#include <memory>

using namespace mlir;

//===----------------------------------------------------------------------===//
// VCalcToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {
// Lower the vcalc.print op to a simple printf
class PrintOpLowering : public ConversionPattern {
  public:
    explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(vcalc::PrintOp::getOperationName(), 1, context) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value>, ConversionPatternRewriter &rewriter) const override {
      auto *context = rewriter.getContext();
      auto loc = op->getLoc();

      ModuleOp parentModule = op->getParentOfType<ModuleOp>();

      auto printRef = getOrInsertPrintf(rewriter, parentModule);
      Value formatSpecifier = getOrCreateGlobalString(
	  loc, rewriter, "fmt_spec", StringRef("Hello 429!\n\0", 12), parentModule);

      rewriter.create<LLVM::CallOp>(
	loc, getPrintfType(context), printRef,
	ArrayRef<Value>({formatSpecifier})
      );
      rewriter.eraseOp(op);
      return success();
    }

  private:
    // Create the function declaration for printf
    static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
      auto i32Type = IntegerType::get(context, 32);
      auto ptrType = LLVM::LLVMPointerType::get(context);
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, ptrType, /*isVarArg=*/true);

      return funcType;
    }
    
    // return the symbol reference to the printf function, if necessary
    static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter, ModuleOp module) {
      auto *context = module.getContext();
      if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
	return SymbolRefAttr::get(context, "printf");

      // insert it since it was not found
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", getPrintfType(context));
      return SymbolRefAttr::get(context, "printf");
    }

    // return a value representing an access into a global string with the given name
    static Value getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name, StringRef value, ModuleOp module) {
      LLVM::GlobalOp global;
      if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
	OpBuilder::InsertionGuard insertGuard(builder);
	builder.setInsertionPointToStart(module.getBody());
	auto type = LLVM::LLVMArrayType::get(
	    IntegerType::get(builder.getContext(), 8), value.size());
	global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant*/ true, LLVM::Linkage::Internal, name, builder.getStringAttr(value), /*alignment*/0);
      }

      // get the pointer to the first character
      Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
      Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),builder.getIndexAttr(0));
      return builder.create<LLVM::GEPOp>(
	  loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(), globalPtr, ArrayRef<Value>({cst0, cst0}));
    }
};
}


//===----------------------------------------------------------------------===//
// VCalcToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
// CRTP pass definition
struct VCalcToLLVMLoweringPass : public PassWrapper<VCalcToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VCalcToLLVMLoweringPass)
  StringRef getArgument() const override { return "vcalc-to-llvm"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() final;
};
}

void VCalcToLLVMLoweringPass::runOnOperation() {
  // define the conversion target
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  // Get the rewrite pattern set
  // If you need to lower other dialects, you can add them here
  RewritePatternSet patterns(&getContext());

  // populateAffineToStdConversionPatterns(patterns);
  // populateSCFToControlFlowConversionPatterns(patterns);
  // ...
  
  patterns.add<PrintOpLowering>(&getContext());

  // since we want to fully convert to LLVM, we can declare a full rewrite
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::vcalc::createLowerToLLVMPass() {
  return std::make_unique<VCalcToLLVMLoweringPass>();
}
