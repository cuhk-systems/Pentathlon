#ifndef ALLOCATE_PASS_H
#define ALLOCATE_PASS_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createDisaggAllocPass();

#endif // ALLOCATE_PASS_H