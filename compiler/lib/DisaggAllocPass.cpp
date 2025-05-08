#include "DisaggAllocPass.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace {
struct DisaggAllocPass : public PassWrapper<DisaggAllocPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    LLVM::LLVMFuncOp disaggAllocFunc = ensureDisaggAllocFunc(module);

    module.walk([&](LLVM::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (!callee || *callee != "malloc")
        return;

      // Expecting malloc(size_t), i.e., i64
      Value mallocSize = callOp.getOperand(0);
      OpBuilder builder(callOp);
      auto newCall = builder.create<LLVM::CallOp>(
          callOp.getLoc(), disaggAllocFunc, ValueRange{mallocSize});

      callOp.replaceAllUsesWith(newCall.getResult());
      callOp.erase();
    });
  }

  LLVM::LLVMFuncOp ensureDisaggAllocFunc(ModuleOp module) {
    auto *ctx = module.getContext();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidPtrTy = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    auto funcTy = LLVM::LLVMFunctionType::get(voidPtrTy, {i64Ty}, false);

    if (auto f = module.lookupSymbol<LLVM::LLVMFuncOp>("disagg_alloc"))
      return f;

    OpBuilder builder(module.getBodyRegion());
    return builder.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "disagg_alloc", funcTy);
  }

  StringRef getArgument() const final { return "allocate-pass"; }
  StringRef getDescription() const final {
    return "Replace malloc calls with disagg_alloc.";
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createDisaggAllocPass() {
  return std::make_unique<DisaggAllocPass>();
}

/// Register the pass
static PassRegistration<DisaggAllocPass> pass;
