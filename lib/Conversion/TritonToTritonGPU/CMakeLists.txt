add_mlir_conversion_library(TritonToTritonGPU
    TritonToTritonGPUPass.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonToTritonGPU
    ${PROJECT_BINARY_DIR}/include/triton/Conversion/TritonToTritonGPU
    ${LLVMGenXIntrinscs_LIB_DIR}

    DEPENDS
    TritonConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
)
