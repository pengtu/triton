add_triton_ut(
	NAME TestSwizzling
	SRCS SwizzleTest.cpp
	LIBS TritonGPUIR TritonTransforms ${dialect_libs} ${conversion_libs}
)
