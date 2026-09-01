llvm_map_components_to_libnames(LLVM_TOOL_LIBS
  Core
  IRReader
  Support
)

add_llvm_executable(my-llvm-opt main.cpp)

llvm_update_compile_flags(my-llvm-opt)

target_include_directories(my-llvm-opt
  PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(my-llvm-opt
  PRIVATE
    CustomLLVMPasses
    ${LLVM_TOOL_LIBS}
)

set_target_properties(my-llvm-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
