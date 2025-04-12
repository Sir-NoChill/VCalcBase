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

int BackEnd::emitLLVM() {  
    // The only remaining dialects in our module after the passes are builtin
    // and LLVM. Setup translation patterns to get them to LLVM IR.
    mlir::registerBuiltinDialectTranslation(this->context);
    mlir::registerLLVMDialectTranslation(this->context);

    llvm::LLVMContext llvm_context;
    auto llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);

    // Create llvm ostream and dump into the output file
    // llvm::verifyModule(*llvm_module);
    llvm_module->dump();
    return 0;
    // return llvm_module;
}

int BackEnd::emitBinary(const char* filename) {
  if (!module) std::cerr << "No llvm_module when lowering to binary" << std::endl;

  llvm::LLVMContext llvm_context;

  mlir::registerBuiltinDialectTranslation(this->context);
  mlir::registerLLVMDialectTranslation(this->context);
  auto llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);

  // Taken from the llc.cpp file
  llvm::Triple triple;
  const llvm::Target *target = nullptr;
  std::unique_ptr<llvm::TargetMachine> target_machine;

  // get the default triple
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllTargetMCAs();
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  triple.setTriple(llvm::sys::getDefaultTargetTriple());

  // get the target
  std::string error;
  target = llvm::TargetRegistry::lookupTarget(llvm::codegen::getMArch(), triple, error);
  if (!target) {
    std::cerr << "No target found: " << error << std::endl;
    exit(1);
  }
  std::string cpu_string = llvm::codegen::getCPUStr(),
	      features_string = llvm::codegen::getFeaturesStr();
  llvm::CodeGenOptLevel lvl = llvm::CodeGenOptLevel::Aggressive;
  llvm::TargetOptions opts = llvm::codegen::InitTargetOptionsFromCodeGenFlags(triple);
  // Needed to link to libc unless we compile with full PIE
  auto RM = std::make_optional(llvm::Reloc::Model::DynamicNoPIC);
  // std::optional<llvm::Reloc::Model> RM = llvm::codegen::getExplicitRelocModel();
  std::optional<llvm::CodeModel::Model> CM = llvm::codegen::getExplicitCodeModel();

  llvm::PassInstrumentationCallbacks pic;
  llvm::TargetLibraryInfoImpl tlii((triple));

  target_machine = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
          triple.getTriple(), cpu_string, features_string, opts, RM, CM, lvl));
  llvm_module->setDataLayout(target_machine->createDataLayout());

  //////////////// If you want to do some LLVMIR Passes, you could do them here ////
  // llvm::MachineModuleInfo mmi(target_machine.get());
  // llvm::MachineFunctionAnalysisManager mfam;
  // llvm::LoopAnalysisManager lam;
  // llvm::FunctionAnalysisManager fam;
  // llvm::CGSCCAnalysisManager cgam;
  // llvm::ModuleAnalysisManager mam;
  // llvm::PassBuilder pb(target_machine.get(), 
  //                      llvm::PipelineTuningOptions(), 
  //                      std::nullopt, nullptr);
  //
  // pb.registerModuleAnalyses(mam);
  // pb.registerCGSCCAnalyses(cgam);
  // pb.registerFunctionAnalyses(fam);
  // pb.registerLoopAnalyses(lam);
  // pb.registerMachineFunctionAnalyses(mfam);
  // pb.crossRegisterProxies(lam, fam, cgam, mam, &mfam);
  //
  // fam.registerPass([&] { return llvm::TargetLibraryAnalysis(tlii); });
  // mam.registerPass([&] { return llvm::MachineModuleAnalysis(mmi); });
  //
  // llvm::ModulePassManager mpm;
  // llvm::FunctionPassManager fpm;
  //
  std::error_code EC;
  llvm::raw_fd_ostream errstream("test.err", EC);
  std::string outfile = filename; // std::tmpnam(nullptr);
  auto out = std::make_unique<llvm::ToolOutputFile>(outfile, EC,
                                               llvm::sys::fs::OF_None);
  llvm::raw_pwrite_stream *outstream = &out->os();

  // auto err = target_machine->buildCodeGenPipeline(
  //   mpm, out->os(), &errstream, 
  //   llvm::CodeGenFileType::ObjectFile, 
  //   llvm::getCGPassBuilderOption(), nullptr);
  // if (err) exit(1);
  // auto pa = mpm.run(*llvm_module, mam);
  //
  // if (llvm_context.getDiagHandlerPtr()->HasErrors) 
  //   std::cerr << "Diagnostic errors";
  
  /////////////// Need to use the legacy pass manager for object lowering /////
  llvm::legacy::PassManager pm;
  pm.add(new llvm::TargetLibraryInfoWrapperPass(tlii));

  auto *MMIWP =
        new llvm::MachineModuleInfoWrapperPass(target_machine.get());
  target_machine->addPassesToEmitFile(
      pm, *outstream, &errstream, llvm::CodeGenFileType::ObjectFile);
  const_cast<llvm::TargetLoweringObjectFile *>
    (target_machine->getObjFileLowering())
    ->Initialize(MMIWP->getMMI().getContext(), *target_machine);

  pm.run(*llvm_module);

  out->keep();

  return 0;
}

