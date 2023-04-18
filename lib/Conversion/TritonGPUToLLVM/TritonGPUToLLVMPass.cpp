#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM//ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVM.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetPlatform.hpp"

#include "ConvertLayoutOpToLLVM.h"
#include "DotOpToLLVM.h"
#include "ElementwiseOpToLLVM.h"
#include "LoadStoreOpToLLVM.h"
#include "ReduceOpToLLVM.h"
#include "TritonGPUToLLVM.h"
#include "TypeConverter.h"
#include "ViewOpToLLVM.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx, bool isROCM, bool isSPIRV)
      : ConversionTarget(ctx) {
    addLegalDialect<index::IndexDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    if (isROCM) {
      addLegalDialect<ROCDL::ROCDLDialect>();
    } else if (isSPIRV) {
      addIllegalDialect<mlir::spirv::SPIRVDialect>();
    } else {
      addLegalDialect<NVVM::NVVMDialect>();
    }
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ReturnOpConversion : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern<triton::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op.getNumOperands();

    // Currently, Triton kernel function always return nothing.
    // TODO(Superjomn) add support for non-inline device function
    if (numArguments > 0) {
      return rewriter.notifyMatchFailure(
          op, "Only kernel function with nothing returned is supported.");
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                op->getAttrs());
    return success();
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(LLVMTypeConverter &converter, int numWarps,
                   PatternBenefit benefit)
      : FuncOpConversionBase(converter, benefit), numWarps(numWarps) {}

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (!newFuncOp) {
      return failure();
    }

    auto ctx = funcOp->getContext();

    // Set an attribute to indicate this function is a kernel entry.
    newFuncOp->setAttr("nvvm.kernel",
                       rewriter.getIntegerAttr(type::u1Ty(ctx), 1));

    // Set an attribute for maxntidx, it could be used in latter LLVM codegen
    // for `nvvm.annotation` metadata.
    newFuncOp->setAttr("nvvm.maxntid", rewriter.getI32ArrayAttr(32 * numWarps));
    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  int numWarps{0};
};

struct FuncOpToSPIRVConversionBase : public OpConversionPattern<triton::FuncOp> {
protected:
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  spirv::FuncOp
  convertFuncOpToSPIRVFuncOp(triton::FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const {

    return nullptr;
  }

};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpToSPIRVConversion : public FuncOpToSPIRVConversionBase {
  FuncOpToSPIRVConversion(SPIRVTypeConverter &converter, MLIRContext *context, int numWarps,
                   PatternBenefit benefit)
          : FuncOpToSPIRVConversionBase(converter, context, benefit), NumWarps(numWarps) {}

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp mod = dyn_cast<ModuleOp>(funcOp->getParentOp());
    if (!mod)
      return failure();

    auto fnType = funcOp.getFunctionType();
    if (fnType.getNumResults() > 1)
      return failure();

    int num_inputs = fnType.getNumInputs();
    TypeConverter::SignatureConversion signatureConverter(num_inputs);
    for (const auto &argType : enumerate(fnType.getInputs())) {
      auto convertedType = getTypeConverter()->convertType(argType.value());
      if (!convertedType)
        return failure();
      signatureConverter.addInputs(argType.index(), convertedType);
    }

    Type resultType;
    if (fnType.getNumResults() == 1) {
      resultType = getTypeConverter()->convertType(fnType.getResult(0));
      if (!resultType)
        return failure();
    }

    // Create the converted spv.func op.
    auto newFuncOp = rewriter.create<spirv::FuncOp>(
            funcOp.getLoc(), funcOp.getName(),
            rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                     resultType ? TypeRange(resultType)
                                                : TypeRange()));

    // Set the SPIRV kernel entry point
    newFuncOp->setAttr(spirv::getEntryPointABIAttrName(), 
      spirv::EntryPointABIAttr::get(getContext(), nullptr, std::nullopt));

    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : funcOp->getAttrs()) {
        if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
            namedAttr.getName() != SymbolTable::getSymbolAttrName() &&
            namedAttr.getName() != funcOp.getArgAttrsAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    ArrayAttr attrs = funcOp.getAllArgAttrs();
    for(int i = 0; i < attrs.size(); i++) {
      if (attrs[i].isa<mlir::DictionaryAttr>()) {
        newFuncOp.setArgAttrs(i, attrs[i].dyn_cast<mlir::DictionaryAttr>());
      }
    }

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(
            &newFuncOp.getBody(), *getTypeConverter(), &signatureConverter)))
      return failure();
    rewriter.eraseOp(funcOp);
    return success();

  }

private:
  int NumWarps{0};
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx, bool isROCM, bool isSPIRV)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    if (isROCM) {
      addLegalDialect<ROCDL::ROCDLDialect>();
    } else if (isSPIRV) {
      addIllegalDialect<spirv::SPIRVDialect>();
    } else {
      addLegalDialect<NVVM::NVVMDialect>();
    }
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class ConvertTritonGPUToLLVM
    : public ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {

public:
  explicit ConvertTritonGPUToLLVM(int computeCapability, bool isROCM, bool isSPIRV)
      : computeCapability(computeCapability), isROCM(isROCM), isSPIRV(isSPIRV) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget target(*context, isROCM, isSPIRV);
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    /* preprocess */
    decomposeMmaToDotOperand(mod, numWarps);
    decomposeBlockedToDotOperand(mod);
    if (failed(decomposeInsertSliceAsyncOp(mod)))
      return signalPassFailure();

    /* allocate shared memory and set barrier */
    Allocation allocation(mod);
    MembarAnalysis membarPass(&allocation);
    membarPass.run();

    /* lower functions */
    {
      mlir::LowerToLLVMOptions option(context);
      TritonGPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context, isROCM, isSPIRV);
      RewritePatternSet funcPatterns(context);
      if (!isSPIRV) {
        funcPatterns.add<FuncOpConversion>(typeConverter, numWarps,
                                           /*benefit=*/1);
      } else {
        auto triple = spirv::VerCapExtAttr::get(
            spirv::Version::V_1_4, {spirv::Capability::Kernel},
            ArrayRef<spirv::Extension>(), context);
        auto targetAttr = spirv::TargetEnvAttr::get(
            triple, spirv::getDefaultResourceLimits(context),
            spirv::ClientAPI::OpenCL, spirv::Vendor::Unknown,
            spirv::DeviceType::Unknown, spirv::TargetEnvAttr::kUnknownDeviceID);
        SPIRVConversionOptions options;
        mod->setAttr(spirv::getTargetEnvAttrName(), targetAttr);
        TritonGPUToSPIRVTypeConverter spirvTypeConverter(targetAttr, options);
        funcPatterns.add<FuncOpToSPIRVConversion>(spirvTypeConverter, context,
                                                  numWarps, /*benefit=*/ 1);
        mlir::populateSPIRVToLLVMTypeConversion(typeConverter);
        populateSPIRVToLLVMFunctionConversionPatterns(typeConverter, funcPatterns);
      }
      funcPatterns.add<ReturnOpConversion>(typeConverter);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // mod->dump();

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AxisInfoAnalysis *axisInfoAnalysis = solver->load<AxisInfoAnalysis>();
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();
    initSharedMemory(allocation.getSharedMemorySize(), typeConverter);
    mod->setAttr("triton_gpu.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32),
                                        allocation.getSharedMemorySize()));

    /* rewrite ops */
    RewritePatternSet patterns(context);
    // TritonGPU lowering patterns
    OpBuilder::InsertPoint indexInsertPoint;
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo indexCacheInfo{
        &baseIndexCache, &indexCache, &indexInsertPoint};
    auto populatePatterns1 = [&](auto populateFunc) {
      populateFunc(typeConverter, patterns, numWarps, *axisInfoAnalysis,
                   &allocation, smem, indexCacheInfo, /*benefit*/ 1);
    };
    auto populatePatterns2 = [&](auto populateFunc) {
      populateFunc(typeConverter, patterns, numWarps, *axisInfoAnalysis,
                   &allocation, smem, /*benefit*/ 1);
    };
    populatePatterns1(populateTritonGPUToLLVMPatterns);
    populatePatterns1(populateConvertLayoutOpToLLVMPatterns);
    populatePatterns2(populateDotOpToLLVMPatterns);
    populatePatterns2(populateElementwiseOpToLLVMPatterns);
    populatePatterns1(populateLoadStoreOpToLLVMPatterns);
    populatePatterns1(populateReduceOpToLLVMPatterns);
    populatePatterns2(populateViewOpToLLVMPatterns);

    // Native lowering patterns
    if (isROCM) {
      mlir::populateGpuToROCDLConversionPatterns(typeConverter, patterns,
                                                 mlir::gpu::amd::HIP);
    } else if (isSPIRV) {
      auto triple = spirv::VerCapExtAttr::get(
          spirv::Version::V_1_4, {spirv::Capability::Kernel},
          ArrayRef<spirv::Extension>(), context);
      auto targetAttr = spirv::TargetEnvAttr::get(
          triple, spirv::getDefaultResourceLimits(context),
          spirv::ClientAPI::OpenCL, spirv::Vendor::Unknown,
          spirv::DeviceType::Unknown, spirv::TargetEnvAttr::kUnknownDeviceID);
      SPIRVConversionOptions options;
      mod->setAttr(spirv::getTargetEnvAttrName(), targetAttr);

      TritonGPUToSPIRVTypeConverter spirvTypeConverter(targetAttr, options);
      mlir::populateGPUToSPIRVPatterns(spirvTypeConverter, patterns);
      mlir::populateSPIRVToLLVMTypeConversion(typeConverter);
      mlir::populateSPIRVToLLVMConversionPatterns(typeConverter, patterns);
    } else {
      mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
    }

    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }

private:
  Value smem;

  using IndexCacheKeyT = std::pair<Attribute, RankedTensorType>;
  DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
      baseIndexCache;
  DenseMap<IndexCacheKeyT, SmallVector<SmallVector<Value>>,
           CacheKeyDenseMapInfo>
      indexCache;

  int computeCapability{};
  bool isROCM{};
  bool isSPIRV{};

  void initSharedMemory(size_t size,
                        TritonGPUToLLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/0,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::kSharedMemorySpace));
    SmallVector<LLVM::LLVMFuncOp> funcs;
    mod.walk([&](LLVM::LLVMFuncOp func) { funcs.push_back(func); });
    assert(funcs.size() == 1 &&
           "Inliner pass is expected before TritonGPUToLLVM");
    b.setInsertionPointToStart(&funcs[0].getBody().front());
    smem = b.create<LLVM::AddressOfOp>(loc, global);
    auto ptrTy =
        LLVM::LLVMPointerType::get(typeConverter.convertType(b.getI8Type()), 3);
    smem = b.create<LLVM::BitcastOp>(loc, ptrTy, smem);
  }

  void decomposeMmaToDotOperand(ModuleOp mod, int numWarps) const {
    // Replace `mma -> dot_op` with `mma -> blocked -> dot_op`
    // unless certain conditions are met
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcMma =
          srcType.getEncoding().dyn_cast<triton::gpu::MmaEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcMma && dstDotOp && !isMmaToDotShortcut(srcType, dstType)) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), srcType.getShape(), getSizePerThread(srcMma),
                getOrder(srcMma), numWarps));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  void decomposeBlockedToDotOperand(ModuleOp mod) const {
    // Replace `blocked -> dot_op` with `blocked -> shared -> dot_op`
    // because the codegen doesn't handle `blocked -> dot_op` directly
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcBlocked && dstDotOp) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::SharedEncodingAttr::get(
                mod.getContext(), dstDotOp, srcType.getShape(),
                getOrder(srcBlocked), srcType.getElementType()));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  LogicalResult decomposeInsertSliceAsyncOp(ModuleOp mod) const {
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AxisInfoAnalysis *axisInfoAnalysis = solver->load<AxisInfoAnalysis>();
    if (failed(solver->initializeAndRun(mod)))
      return failure();
    // TODO(Keren): This is a hacky knob that may cause performance regression
    // when decomposition has been performed. We should remove this knob once we
    // have thorough analysis on async wait. Currently, we decompose
    // `insert_slice_async` into `load` and `insert_slice` without knowing which
    // `async_wait` is responsible for the `insert_slice_async`. To guarantee
    // correctness, we blindly set the `async_wait` to wait for all async ops.
    //
    // There are two options to improve this:
    // 1. We can perform a dataflow analysis to find the `async_wait` that is
    // responsible for the `insert_slice_async` in the backend.
    // 2. We can modify the pipeline to perform the decomposition before the
    // `async_wait` is inserted. However, it is also risky because we don't know
    // the correct vectorized shape yet in the pipeline pass. Making the
    // pipeline pass aware of the vectorization could introduce additional
    // dependencies on the AxisInfoAnalysis and the Coalesce analysis.
    bool decomposed = false;
    // insert_slice_async %src, %dst, %idx, %mask, %other
    // =>
    // %tmp = load %src, %mask, %other
    // %res = insert_slice %tmp into %dst[%idx]
    mod.walk([&](triton::gpu::InsertSliceAsyncOp insertSliceAsyncOp) -> void {
      OpBuilder builder(insertSliceAsyncOp);

      // Get the vectorized load size
      auto src = insertSliceAsyncOp.getSrc();
      auto dst = insertSliceAsyncOp.getDst();
      auto srcTy = src.getType().cast<RankedTensorType>();
      auto dstTy = dst.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcTy.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto resSharedLayout =
          dstTy.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      auto resElemTy = dstTy.getElementType();
      unsigned inVec = axisInfoAnalysis->getPtrContiguity(src);
      unsigned outVec = resSharedLayout.getVec();
      unsigned minVec = std::min(outVec, inVec);
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto byteWidth = bitWidth / 8;

      // If the load byte width is not eligible or the current compute
      // capability does not support async copy, then we do decompose
      if (triton::gpu::InsertSliceAsyncOp::getEligibleLoadByteWidth(
              computeCapability)
              .contains(byteWidth))
        return;

      // load
      auto tmpTy =
          RankedTensorType::get(srcTy.getShape(), resElemTy, srcBlocked);
      auto loadOp = builder.create<triton::LoadOp>(
          insertSliceAsyncOp.getLoc(), tmpTy, insertSliceAsyncOp.getSrc(),
          insertSliceAsyncOp.getMask(), insertSliceAsyncOp.getOther(),
          // TODO(Chenggang): confirm `boundaryCheck` and `padding`
          /*boundaryCheck=*/nullptr, /*padding=*/nullptr,
          insertSliceAsyncOp.getCache(), insertSliceAsyncOp.getEvict(),
          insertSliceAsyncOp.getIsVolatile());

      // insert_slice
      auto axis = insertSliceAsyncOp.getAxis();
      auto intAttr = [&](int64_t v) { return builder.getI64IntegerAttr(v); };
      auto offsets = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(0));
      auto sizes = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      auto strides = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      offsets[axis] = insertSliceAsyncOp.getIndex();
      for (size_t i = 0; i < dstTy.getRank(); i++) {
        if (i != axis)
          sizes[i] = intAttr(dstTy.getShape()[i]);
      }
      auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
          insertSliceAsyncOp.getLoc(), loadOp, insertSliceAsyncOp.getDst(),
          offsets, sizes, strides);

      // Replace
      insertSliceAsyncOp.replaceAllUsesWith(insertSliceOp.getResult());
      insertSliceAsyncOp.erase();
      decomposed = true;
    });

    mod.walk([&](triton::gpu::AsyncCommitGroupOp asyncCommitGroupOp) -> void {
      if (!triton::gpu::AsyncCommitGroupOp::isSupported(computeCapability))
        asyncCommitGroupOp.erase();
    });

    mod.walk([&](triton::gpu::AsyncWaitOp asyncWaitOp) -> void {
      if (!triton::gpu::AsyncWaitOp::isSupported(computeCapability)) {
        // async wait is supported in Ampere and later
        asyncWaitOp.erase();
      } else if (decomposed) {
        // Wait for all previous async ops
        OpBuilder builder(asyncWaitOp);
        builder.create<triton::gpu::AsyncWaitOp>(asyncWaitOp.getLoc(), 0);
        asyncWaitOp.erase();
      }
    });
    return success();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int computeCapability, bool isROCM, bool isSPIRV) {
  return std::make_unique<::ConvertTritonGPUToLLVM>(computeCapability, isROCM, isSPIRV);
}

} // namespace triton
} // namespace mlir
