#include "BackEnd.h"

BackEnd::BackEnd() : loc(mlir::UnknownLoc::get(&context)) {
    // Load Dialects.
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    // context.loadDialect<mlir::arith::ArithDialect>();
    // context.loadDialect<mlir::scf::SCFDialect>();
    // context.loadDialect<mlir::cf::ControlFlowDialect>();
    // context.loadDialect<mlir::memref::MemRefDialect>(); 
    context.getOrLoadDialect<mlir::vcalc::VCalcDialect>();

    // Initialize the MLIR context 
    builder = std::make_shared<mlir::OpBuilder>(&context);
    module = mlir::ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
}

int BackEnd::emitModule() {

    // Create a main function 
    mlir::Type intType = mlir::IntegerType::get(&context, 32);
    auto mainType = mlir::LLVM::LLVMFunctionType::get(intType, {}, false);
    mlir::LLVM::LLVMFuncOp mainFunc = builder->create<mlir::LLVM::LLVMFuncOp>(loc, "main", mainType);
    mlir::Block *entry = mainFunc.addEntryBlock(*builder);
    builder->setInsertionPointToStart(entry);

    // Example print op
    builder->create<mlir::vcalc::PrintOp>(loc);

    // Return 0
    mlir::Value zero = builder->create<mlir::LLVM::ConstantOp>(loc, intType, builder->getIntegerAttr(intType, 0));
    builder->create<mlir::LLVM::ReturnOp>(builder->getUnknownLoc(), zero);    
    
    module.dump(); // show the module before translation

    if (mlir::failed(mlir::verify(module))) {
        module.emitError("module failed to verify");
        return 1;
    }
    return 0;
}

int BackEnd::lowerDialects() {
    // Set up the MLIR pass manager to iteratively lower all the Ops
    mlir::PassManager pm(&context);

    // Lower SCF to CF (ControlFlow)
    // pm.addPass(mlir::createConvertSCFToCFPass());

    // Lower Arith to LLVM
    // pm.addPass(mlir::createArithToLLVMConversionPass());

    // Lower MemRef to LLVM
    // pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

    // Lower CF to LLVM
    // pm.addPass(mlir::createConvertControlFlowToLLVMPass());

    // Finalize the conversion to LLVM dialect
    // pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    
    pm.addPass(mlir::vcalc::createLowerToLLVMPass());


    // Run the passes
    if (mlir::failed(pm.run(module))) {
        llvm::errs() << "Pass pipeline failed\n";
        return 1;
    }
    module.dump(); // show the module after translation
    return 0;
}

void BackEnd::dumpLLVM(std::ostream &os) {  
    // // Initialize LLVM targets.
    // // If you want to generate an executable
    // llvm::InitializeNativeTarget();
    // llvm::InitializeNativeTargetAsmPrinter();

    // The only remaining dialects in our module after the passes are builtin
    // and LLVM. Setup translation patterns to get them to LLVM IR.
    mlir::registerBuiltinDialectTranslation(this->context);
    mlir::registerLLVMDialectTranslation(this->context);

    llvm::LLVMContext llvm_context;
    auto llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);

    // Create llvm ostream and dump into the output file
    llvm::raw_os_ostream output(os);
    output << *llvm_module; // Dump the fully converted LLVMIR module
}

