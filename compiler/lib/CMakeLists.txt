add_llvm_executable(my-opt main.cpp)

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

target_link_libraries(my-opt
  PRIVATE
  CustomMLIRPasses
  MLIRParser
  MLIRIR
  MLIRLLVMIR
  MLIRSupport
  MLIRPass
  MLIRTransforms
)
