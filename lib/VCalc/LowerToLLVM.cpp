#include "VCalc/Ops.h"
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
class VCalcTypeConverter : public TypeConverter {
  public:
    // The identity type conversion
    // TODO add more as required
    // https://mlir.llvm.org/docs/DialectConversion/#type-conversion
    VCalcTypeConverter(MLIRContext *context) {
      addConversion([](Type type) {return type; });
    }
};
}

namespace {
// Lower the vcalc.print op to a simple printf
class PrintOpLowering : public OpConversionPattern<mlir::vcalc::PrintOp> {
  using OpConversionPattern<mlir::vcalc::PrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::vcalc::PrintOp op, OpAdaptor, 
      ConversionPatternRewriter &rewriter) const override {
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

namespace mlir {
namespace vcalc {

/// This defines the whole pass from tablegen
#define GEN_PASS_DEF_VCALCTOMLIR
#include "VCalc/Passes.h.inc"

// Now implement the pass
struct VCalcToStandard : impl::VCalcToMLIRBase<VCalcToStandard> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto* module = getOperation();
  
    // define the conversion target
    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<VCalcDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    // Get the rewrite pattern set
    // If you need to lower other dialects, you can add them here
    RewritePatternSet patterns(context);
    VCalcTypeConverter vcalcConverter(context);
  
    patterns.add<PrintOpLowering>(vcalcConverter, context);

    // populateAffineToStdConversionPatterns(patterns);
    // populateSCFToControlFlowConversionPatterns(patterns);
    // ...
    
    // since we want to fully convert to LLVM, we can declare a full rewrite
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
}
}
