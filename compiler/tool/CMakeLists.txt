get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(my-opt main.cpp)

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

mlir_check_all_link_libraries(tutorial-opt)