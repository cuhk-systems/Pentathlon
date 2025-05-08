#include "AddrDepPass.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace {
struct AddrDepPass : public PassWrapper<AddrDepPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    LLVM::LLVMFuncOp addAddrDepFunc = ensureAddAddrDepFunc(module);

    module.walk([&](LLVM::StoreOp storeOp) {
      Value value = storeOp.getValue();
      Value addr = storeOp.getAddr();

      // Check both operands are pointers
      if (!value.getType().isa<LLVM::LLVMPointerType>())
        return;
      if (!addr.getType().isa<LLVM::LLVMPointerType>())
        return;

      OpBuilder builder(storeOp);
      builder.create<LLVM::CallOp>(storeOp.getLoc(), addAddrDepFunc, ValueRange{value, addr});
    });
  }

  LLVM::LLVMFuncOp ensureAddAddrDepFunc(ModuleOp module) {
    auto *ctx = module.getContext();
    auto voidPtrTy = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, voidPtrTy}, false);

    if (auto f = module.lookupSymbol<LLVM::LLVMFuncOp>("addAddrDep"))
      return f;

    OpBuilder builder(module.getBodyRegion());
    return builder.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "addAddrDep", funcTy);
  }

  StringRef getArgument() const final { return "addr-dep-pass"; }
  StringRef getDescription() const final {
    return "Insert addAddrDep(value, addr) for pointer stores.";
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createAddrDepPass() {
  return std::make_unique<AddrDepPass>();
}

/// Register the pass
static PassRegistration<AddrDepPass> pass;
