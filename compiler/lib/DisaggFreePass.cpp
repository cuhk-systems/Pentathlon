#include "DisaggFreePass.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace {
struct DisaggFreePass : public PassWrapper<DisaggFreePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    LLVM::LLVMFuncOp disaggFreeFunc = ensureDisaggFreeFunc(module);

    module.walk([&](LLVM::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (!callee || *callee != "free")
        return;

      Value freePtr = callOp.getOperand(0);
      OpBuilder builder(callOp);
      builder.create<LLVM::CallOp>(callOp.getLoc(), disaggFreeFunc, ValueRange{freePtr});

      callOp.erase();
    });
  }

  LLVM::LLVMFuncOp ensureDisaggFreeFunc(ModuleOp module) {
    auto *ctx = module.getContext();
    auto voidPtrTy = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    auto funcTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {voidPtrTy}, false);

    if (auto f = module.lookupSymbol<LLVM::LLVMFuncOp>("disagg_free"))
      return f;

    OpBuilder builder(module.getBodyRegion());
    return builder.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "disagg_free", funcTy);
  }

  StringRef getArgument() const final { return "disagg-free-pass"; }
  StringRef getDescription() const final {
    return "Replace free calls with disagg_free.";
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createDisaggFreePass() {
  return std::make_unique<DisaggFreePass>();
}

/// Register the pass
static PassRegistration<DisaggFreePass> pass;
