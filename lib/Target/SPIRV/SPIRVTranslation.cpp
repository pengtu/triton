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

class SmallVectorBuffer : public std::streambuf
{
  // All memory management is delegated to llvm::SmallVectorImpl
  llvm::SmallVectorImpl<char> &OS;

  // Since we don't touch any pointer in streambuf(pbase, pptr, epptr) this is
  // the only method we need to override.
  virtual std::streamsize xsputn(const char *s, std::streamsize  n) override {
    OS.append(s, s + n);
    return n;
  }

public:
  SmallVectorBuffer() = delete;
  SmallVectorBuffer(const SmallVectorBuffer&) = delete;
  SmallVectorBuffer &operator=(const SmallVectorBuffer&) = delete;
  SmallVectorBuffer(llvm::SmallVectorImpl<char> &O) : OS(O) {}
};

std::string translateLLVMIRToSPIRV(llvm::Module &module) {
  // initLLVM();

  llvm::SmallVector<char, 0> buffer;
  std::string triple = "spir64-unknown-unknown";
  std::string layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64";

  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(module);
  // module->print(llvm::outs(), nullptr);
    
  module.setTargetTriple(triple);
  module.setDataLayout(layout);
  if (module.materializeAll()) {
    llvm::errs() << "SPIRVTranslation: failed to read the LLVM module IR!";
    llvm::errs().flush();
    std::string result(buffer.begin(), buffer.end());
    return result;
  }

  // emit
  SmallVectorBuffer StreamBuf(buffer);
  std::ostream OS(&StreamBuf);
  std::string Err;

  SPIRV::TranslatorOpts SPIRVOpts;
  SPIRVOpts.enableAllExtensions();
  SPIRVOpts.setMemToRegEnabled(true);
  SPIRVOpts.setPreserveOCLKernelArgTypeMetadataThroughString(true);
  auto success = llvm::writeSpirv(&module, SPIRVOpts, OS, Err);

  if (!success) {
    llvm::errs() << "SPIRVTranslation: SPIRV translation failed with"
        << Err.c_str();
    llvm::errs().flush();
  }
  
  std::string result(buffer.begin(), buffer.end());
  return result;
}

} // namespace triton