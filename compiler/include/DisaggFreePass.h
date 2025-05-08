#ifndef DISAGG_FREE_PASS_H
#define DISAGG_FREE_PASS_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createDisaggFreePass();

#endif // DISAGG_FREE_PASS_H
