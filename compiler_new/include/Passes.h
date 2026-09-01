#ifndef PENTATHLON_LLVM_PASSES_H
#define PENTATHLON_LLVM_PASSES_H

#include "llvm/IR/Module.h"

bool runLocalAddrPass(llvm::Module &M);
bool runDisaggAllocPass(llvm::Module &M);
bool runDisaggFreePass(llvm::Module &M);
bool runAddrDepPass(llvm::Module &M);
bool runMarkDirtyPass(llvm::Module &M);
bool runAddrDepRelPass(llvm::Module &M);

#endif
