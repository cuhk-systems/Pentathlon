get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set (LIBS
  ${dialect_libs}
  ${conversion_libs}
  CustomMLIRPasses
  MLIRParser
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIROptLib
  MLIRTransforms
)

add_llvm_executable(my-opt main.cpp)

llvm_update_compile_flags(my-opt)

target_include_directories(my-opt
  PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(my-opt PRIVATE ${LIBS})

set_target_properties(my-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

mlir_check_all_link_libraries(my-opt)