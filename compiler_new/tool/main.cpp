#include "Passes.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string>
    InputFilename(cl::Positional, cl::desc("<input LLVM IR>"), cl::Required);

static cl::opt<std::string>
    OutputFilename("o", cl::desc("Output filename"), cl::init("-"));

static cl::opt<bool> LocalAddrPass("local-addr-pass",
                                   cl::desc("Insert getLocalAddr calls"),
                                   cl::init(false));
static cl::opt<bool> DisaggAllocPass("disagg-alloc-pass",
                                     cl::desc("Replace malloc with disaggAlloc"),
                                     cl::init(false));
static cl::opt<bool> DisaggFreePass("disagg-free-pass",
                                    cl::desc("Replace free with disaggFree"),
                                    cl::init(false));
static cl::opt<bool> AddrDepPass("addr-dep-pass",
                                 cl::desc("Insert addAddrDep after pointer loads"),
                                 cl::init(false));
static cl::opt<bool> MarkDirtyPass("mark-dirty-pass",
                                   cl::desc("Insert markDirty before writes"),
                                   cl::init(false));
static cl::opt<bool> AddrDepRelPass("addr-dep-rel-pass",
                                    cl::desc("Insert updateAddrDep before pointer stores"),
                                    cl::init(false));

int main(int argc, const char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(
      argc, argv, "Pentathlon LLVM IR optimizer (runtime hook insertion)\n");

  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  if (AddrDepPass) {
    runAddrDepPass(*M);
  }
  if (DisaggAllocPass) {
    runDisaggAllocPass(*M);
  }
  if (DisaggFreePass) {
    runDisaggFreePass(*M);
  }
  if (MarkDirtyPass) {
    runMarkDirtyPass(*M);
  }
  if (AddrDepRelPass) {
    runAddrDepRelPass(*M);
  }
  if (LocalAddrPass) {
    runLocalAddrPass(*M);
  }

  std::error_code EC;
  raw_fd_ostream Out(OutputFilename, EC, sys::fs::OF_Text);
  if (EC) {
    errs() << "failed to open output file '" << OutputFilename
           << "': " << EC.message() << "\n";
    return 1;
  }
  M->print(Out, nullptr);
  return 0;
}
