add_executable(my-opt main.cpp)

target_include_directories(my-opt
  PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(my-opt
  PRIVATE
    CustomMLIRPasses
    MLIRParser
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIROptLib
    MLIRTransforms
)

set_target_properties(my-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)