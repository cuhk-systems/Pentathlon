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

target_include_directories(CustomMLIRPasses
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(CustomMLIRPasses
  PUBLIC
    MLIRIR
    MLIRLLVMIR
    MLIRPass
    MLIRSupport
)