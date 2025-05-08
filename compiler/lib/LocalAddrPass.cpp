#include "LocalAddrPass.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace {
struct LocalAddrPass : public PassWrapper<LocalAddrPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    LLVM::LLVMFuncOp getLocalAddrFunc = ensureGetLocalAddrFunc(module);

    module.walk([&](LLVM::LLVMFuncOp func) {
      func.walk([&](Operation *op) {
        OpBuilder builder(op);

        if (auto loadOp = dyn_cast<LLVM::LoadOp>(op)) {
          Value ptr = loadOp.getAddr();
          Value newPtr = insertGetLocalAddrCall(builder, op, getLocalAddrFunc, ptr);
          loadOp.setOperand(0, newPtr);
        } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
          Value ptr = storeOp.getAddr();
          Value newPtr = insertGetLocalAddrCall(builder, op, getLocalAddrFunc, ptr);
          storeOp.setOperand(1, newPtr);
        } else if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
          SmallVector<Value, 4> newOperands;
          for (Value arg : callOp.getOperands()) {
            if (arg.getType().isa<LLVM::LLVMPointerType>()) {
              Value newArg = insertGetLocalAddrCall(builder, op, getLocalAddrFunc, arg);
              newOperands.push_back(newArg);
            } else {
              newOperands.push_back(arg);
            }
          }
          callOp->setOperands(newOperands);
        }
      });
    });
  }

  Value insertGetLocalAddrCall(OpBuilder &builder, Operation *op,
                                LLVM::LLVMFuncOp func,
                                Value ptr) {
    builder.setInsertionPoint(op);
    auto call = builder.create<LLVM::CallOp>(op->getLoc(), func, ValueRange{ptr});
    return call.getResult(0);
  }

  LLVM::LLVMFuncOp ensureGetLocalAddrFunc(ModuleOp module) {
    auto *ctx = module.getContext();
    auto voidPtrTy = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    auto funcTy = LLVM::LLVMFunctionType::get(voidPtrTy, {voidPtrTy}, false);

    if (auto f = module.lookupSymbol<LLVM::LLVMFuncOp>("getLocalAddr"))
      return f;

    OpBuilder builder(module.getBodyRegion());
    return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "getLocalAddr", funcTy);
  }

  StringRef getArgument() const final { return "local-addr-pass"; }
  StringRef getDescription() const final {
    return "Insert getLocalAddr call before pointer usage in LLVM ops.";
  }
};
} // namespace

std::unique_ptr<Pass> createLocalAddrPass() {
  return std::make_unique<LocalAddrPass>();
}

/// Register the pass in static global constructor
static PassRegistration<LocalAddrPass> pass;
