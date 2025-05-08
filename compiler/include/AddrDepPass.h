#ifndef ADDR_DEP_PASS_H
#define ADDR_DEP_PASS_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createAddrDepPass();

#endif // ADDR_DEP_PASS_H
