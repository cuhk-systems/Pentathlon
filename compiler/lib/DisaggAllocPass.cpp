#include "DisaggAllocPass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace {
struct AllocatePass : public PassWrapper<AllocatePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    LLVM::LLVMFuncOp disaggAllocFunc = ensureDisaggAllocFunc(module);

    module.walk([&](LLVM::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (!callee || *callee != "malloc")
        return;

      // Assume malloc takes i64 and returns !llvm.ptr<i8>
      Value mallocSize = callOp.getOperand(0);

      OpBuilder builder(callOp);
      auto newCall = builder.create<LLVM::CallOp>(
          callOp.getLoc(), disaggAllocFunc, ValueRange{mallocSize});

      // Replace all uses of malloc result with new call result
      if (callOp->getNumResults() == 1) {
        callOp->getResult().replaceAllUsesWith(newCall.getResult());
      }

      // Remove original malloc call
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
    return "Replace malloc calls with disagg_alloc(size).";
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createAllocatePass() {
  return std::make_unique<AllocatePass>();
}

static PassRegistration<AllocatePass> pass;
