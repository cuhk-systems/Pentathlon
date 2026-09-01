add_library(CustomLLVMPasses
  Passes.cpp
)

target_include_directories(CustomLLVMPasses
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)

llvm_map_components_to_libnames(LLVM_LIBS
  Core
  Support
)

target_link_libraries(CustomLLVMPasses
  PUBLIC
    ${LLVM_LIBS}
)
