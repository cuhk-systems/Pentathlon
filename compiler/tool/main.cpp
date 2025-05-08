#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "LocalAddrPass.h"
#include "DisaggAllocPass.h"
#include "DisaggFreePass.h"
#include "AddrDepPass.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register core dialects
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  // Register your pass
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLocalAddrPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createDisaggAllocPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createDisaggFreePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAddrDepPass();
  });

  return mlir::MlirOptMain(argc, argv, "My Custom MLIR Opt Tool\n", registry);
}
