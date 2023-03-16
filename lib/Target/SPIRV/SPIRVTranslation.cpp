#include "triton/Target/SPIRV/SPIRVTranslation.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include <optional>

#include "LLVMSPIRVLib/LLVMSPIRVLib.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

namespace triton {

std::string translateLLVMIRToSPIRV(llvm::Module &module) {
  // initLLVM();

  llvm::SmallVector<char, 0> buffer;

  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(module);
  // module->print(llvm::outs(), nullptr);
  if (module.materializeAll()) {
    llvm::errs() << "SPIRVTranslation: failed to read the LLVM module IR!";
    llvm::errs().flash();
    std::string result(buffer.begin(), buffer.end());
    return result;
  }

  // emit
  llvm::raw_svector_ostream stream(buffer);
  std::string Err;

  SPIRV::TranslatorOpts SPIRVOpts;
  SPIRVOpts.enableAllExtensions();
  SPIRVOpts.setMemToRegEnabled(true);
  SPIRVOpts.setPreserveOCLKernelArgTypeMetadataThroughString(true);
  auto success = llvm::writeSpirv(module.get(), SPIRVOpts, stream, Err);

  if (!success) {
    llvm::errs() << "SPIRVTranslation: SPIRV translation failed with"
        << Err.c_str();
    llvm::errs().flush();
  }
  
  std::string result(buffer.begin(), buffer.end());
  return result;
}

} // namespace triton