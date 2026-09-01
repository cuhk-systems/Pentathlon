#include "Passes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"

#include <utility>

using namespace llvm;

namespace {

static Type *getVoidPtrTy(LLVMContext &Ctx) {
  return PointerType::getUnqual(Ctx);
}

static FunctionCallee ensureFn(Module &M, StringRef Name, Type *RetTy,
                               ArrayRef<Type *> ArgTys) {
  FunctionType *FT = FunctionType::get(RetTy, ArgTys, false);
  return M.getOrInsertFunction(Name, FT);
}

static Value *castToVoidPtr(IRBuilder<> &B, Value *V) {
  Type *VoidPtrTy = getVoidPtrTy(B.getContext());
  if (V->getType() == VoidPtrTy) {
    return V;
  }
  return B.CreateBitCast(V, VoidPtrTy);
}

static Value *castFromVoidPtr(IRBuilder<> &B, Value *V, Type *DstTy) {
  if (V->getType() == DstTy) {
    return V;
  }
  return B.CreateBitCast(V, DstTy);
}

static Value *insertGetLocalAddrCall(IRBuilder<> &B, Module &M, Value *Ptr) {
  FunctionCallee GetLocalAddr =
      ensureFn(M, "getLocalAddr", getVoidPtrTy(M.getContext()),
               {getVoidPtrTy(M.getContext())});
  Value *AsVoid = castToVoidPtr(B, Ptr);
  Value *Call = B.CreateCall(GetLocalAddr, {AsVoid});
  return castFromVoidPtr(B, Call, Ptr->getType());
}

static bool isCompilerHook(StringRef Name) {
  static const DenseSet<StringRef> Hooks = {
      "disaggAlloc", "disaggFree", "getLocalAddr",
      "markDirty",   "addAddrDep", "updateAddrDep"};
  return Hooks.count(Name);
}

static bool isOperatorNew(StringRef Name) {
  return Name.starts_with("_Znwm") || Name.starts_with("_Znam");
}

static bool isOperatorDelete(StringRef Name) {
  return Name.starts_with("_ZdlPv") || Name.starts_with("_ZdaPv");
}

static bool isKnownMemoryWriteCall(StringRef Name) {
  return Name == "memset" || Name == "memcpy" || Name == "memmove" ||
         Name == "pth_memset" || Name == "pth_memcpy" ||
         Name == "pth_memmove";
}

static Value *getWritePointerOperand(Instruction &I) {
  if (auto *SI = dyn_cast<StoreInst>(&I)) {
    return SI->getPointerOperand();
  }
  if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
    return RMW->getPointerOperand();
  }
  if (auto *CmpXchg = dyn_cast<AtomicCmpXchgInst>(&I)) {
    return CmpXchg->getPointerOperand();
  }
  if (auto *MI = dyn_cast<MemIntrinsic>(&I)) {
    return MI->getDest();
  }
  if (auto *CB = dyn_cast<CallBase>(&I)) {
    Function *Callee = CB->getCalledFunction();
    if (Callee && isKnownMemoryWriteCall(Callee->getName()) &&
        CB->arg_size() >= 1) {
      return CB->getArgOperand(0);
    }
  }
  return nullptr;
}

} // namespace

bool runLocalAddrPass(Module &M) {
  SmallVector<Instruction *, 128> Worklist;
  for (Function &F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<CallBase>(I) ||
            isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I)) {
          Worklist.push_back(&I);
        }
      }
    }
  }

  bool Changed = false;
  for (Instruction *I : Worklist) {
    if (auto *LI = dyn_cast<LoadInst>(I)) {
      IRBuilder<> B(LI);
      Value *NewPtr = insertGetLocalAddrCall(B, M, LI->getPointerOperand());
      LI->setOperand(LI->getPointerOperandIndex(), NewPtr);
      Changed = true;
      continue;
    }

    if (auto *SI = dyn_cast<StoreInst>(I)) {
      IRBuilder<> B(SI);
      Value *NewPtr = insertGetLocalAddrCall(B, M, SI->getPointerOperand());
      SI->setOperand(SI->getPointerOperandIndex(), NewPtr);
      Changed = true;
      continue;
    }

    if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
      IRBuilder<> B(RMW);
      Value *NewPtr =
          insertGetLocalAddrCall(B, M, RMW->getPointerOperand());
      RMW->setOperand(AtomicRMWInst::getPointerOperandIndex(), NewPtr);
      Changed = true;
      continue;
    }

    if (auto *CmpXchg = dyn_cast<AtomicCmpXchgInst>(I)) {
      IRBuilder<> B(CmpXchg);
      Value *NewPtr =
          insertGetLocalAddrCall(B, M, CmpXchg->getPointerOperand());
      CmpXchg->setOperand(AtomicCmpXchgInst::getPointerOperandIndex(),
                          NewPtr);
      Changed = true;
      continue;
    }

    auto *CB = dyn_cast<CallBase>(I);
    if (!CB) {
      continue;
    }
    Function *Callee = CB->getCalledFunction();
    if (Callee) {
      if (isCompilerHook(Callee->getName())) {
        continue;
      }
      if (!Callee->isDeclaration()) {
        continue;
      }
    }

    bool CallChanged = false;
    for (unsigned ArgIdx = 0; ArgIdx < CB->arg_size(); ++ArgIdx) {
      Value *Arg = CB->getArgOperand(ArgIdx);
      if (!Arg->getType()->isPointerTy()) {
        continue;
      }
      IRBuilder<> B(CB);
      Value *NewArg = insertGetLocalAddrCall(B, M, Arg);
      CB->setArgOperand(ArgIdx, NewArg);
      CallChanged = true;
    }
    Changed |= CallChanged;
  }

  return Changed;
}

bool runDisaggAllocPass(Module &M) {
  SmallVector<CallInst *, 64> Calls;
  for (Function &F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Function *Callee = CI->getCalledFunction();
          if (Callee && (Callee->getName() == "malloc" ||
                         isOperatorNew(Callee->getName()))) {
            Calls.push_back(CI);
          }
        }
      }
    }
  }

  bool Changed = false;
  FunctionCallee DisaggAlloc =
      ensureFn(M, "disaggAlloc", getVoidPtrTy(M.getContext()),
               {Type::getInt64Ty(M.getContext())});

  for (CallInst *CI : Calls) {
    if (CI->arg_size() < 1) {
      continue;
    }
    IRBuilder<> B(CI);
    Value *Call = B.CreateCall(DisaggAlloc, {CI->getArgOperand(0)});
    Value *Cast = castFromVoidPtr(B, Call, CI->getType());
    CI->replaceAllUsesWith(Cast);
    CI->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

bool runDisaggFreePass(Module &M) {
  SmallVector<CallInst *, 64> Calls;
  for (Function &F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Function *Callee = CI->getCalledFunction();
          if (Callee && (Callee->getName() == "free" ||
                         isOperatorDelete(Callee->getName()))) {
            Calls.push_back(CI);
          }
        }
      }
    }
  }

  bool Changed = false;
  FunctionCallee DisaggFree =
      ensureFn(M, "disaggFree", Type::getVoidTy(M.getContext()),
               {getVoidPtrTy(M.getContext())});

  for (CallInst *CI : Calls) {
    if (CI->arg_size() < 1) {
      continue;
    }
    IRBuilder<> B(CI);
    Value *Arg = castToVoidPtr(B, CI->getArgOperand(0));
    B.CreateCall(DisaggFree, {Arg});
    CI->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

bool runAddrDepPass(Module &M) {
  SmallVector<LoadInst *, 64> Loads;
  for (Function &F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (LI->getPointerOperandType()->isPointerTy() &&
              LI->getType()->isPointerTy()) {
            Loads.push_back(LI);
          }
        }
      }
    }
  }

  bool Changed = false;
  FunctionCallee AddAddrDep =
      ensureFn(M, "addAddrDep", Type::getVoidTy(M.getContext()),
               {getVoidPtrTy(M.getContext()), getVoidPtrTy(M.getContext())});

  for (LoadInst *LI : Loads) {
    IRBuilder<> B(LI->getNextNode());
    Value *FromAddr = castToVoidPtr(B, LI->getPointerOperand());
    Value *Res = castToVoidPtr(B, LI);
    B.CreateCall(AddAddrDep, {FromAddr, Res});
    Changed = true;
  }

  return Changed;
}

bool runMarkDirtyPass(Module &M) {
  SmallVector<std::pair<Instruction *, Value *>, 64> Writes;
  for (Function &F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        Value *Ptr = getWritePointerOperand(I);
        if (Ptr && Ptr->getType()->isPointerTy()) {
          Writes.push_back({&I, Ptr});
        }
      }
    }
  }

  bool Changed = false;
  FunctionCallee MarkDirty =
      ensureFn(M, "markDirty", Type::getVoidTy(M.getContext()),
               {getVoidPtrTy(M.getContext())});

  for (auto [I, Ptr] : Writes) {
    IRBuilder<> B(I);
    Value *Addr = castToVoidPtr(B, Ptr);
    B.CreateCall(MarkDirty, {Addr});
    Changed = true;
  }

  return Changed;
}

bool runAddrDepRelPass(Module &M) {
  SmallVector<StoreInst *, 64> Stores;
  for (Function &F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (SI->getPointerOperandType()->isPointerTy() &&
              SI->getValueOperand()->getType()->isPointerTy()) {
            Stores.push_back(SI);
          }
        }
      }
    }
  }

  bool Changed = false;
  FunctionCallee UpdateAddrDep =
      ensureFn(M, "updateAddrDep", getVoidPtrTy(M.getContext()),
               {getVoidPtrTy(M.getContext()), getVoidPtrTy(M.getContext())});

  for (StoreInst *SI : Stores) {
    IRBuilder<> B(SI);
    Value *Father = castToVoidPtr(B, SI->getPointerOperand());
    Value *Child = castToVoidPtr(B, SI->getValueOperand());
    Value *Call = B.CreateCall(UpdateAddrDep, {Father, Child});
    Value *NewVal = castFromVoidPtr(B, Call, SI->getValueOperand()->getType());
    SI->setOperand(0, NewVal);
    Changed = true;
  }

  return Changed;
}
