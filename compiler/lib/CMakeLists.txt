add_mlir_library(CustomMLIRPasses
  LocalAddrPass.cpp
  DisaggAllocPass.cpp
  DisaggFreePass.cpp
  AddrDepPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRSupport
)