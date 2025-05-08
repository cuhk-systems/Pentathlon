#ifndef LOCAL_ADDR_PASS_H
#define LOCAL_ADDR_PASS_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createLocalAddrPass();

#endif // LOCAL_ADDR_PASS_H
