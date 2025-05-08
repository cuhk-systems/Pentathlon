add_mlir_library(CustomMLIRPasses
  LocalAddrPass.cpp
  AllocatePass.cpp
  DisaggFreePass.cpp
  AddrDepPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRSupport
)

add_executable(my-opt main.cpp)

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
