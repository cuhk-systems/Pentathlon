$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$CompilerNewDir = Join-Path $RepoRoot "compiler_new"
$BuildDir = Join-Path $CompilerNewDir "build"
$TestDir = $PSScriptRoot

# Override these if your binaries are elsewhere.
$Clang = if ($env:CLANG_PATH) { $env:CLANG_PATH } else { "clang" }
$MyLlvmOpt = if ($env:MY_LLVM_OPT_PATH) { $env:MY_LLVM_OPT_PATH } else { (Join-Path $BuildDir "bin\\my-llvm-opt") }

$InputC = Join-Path $TestDir "smoke_test.c"
$InputLl = Join-Path $TestDir "smoke_test.ll"
$OutputLl = Join-Path $TestDir "smoke_test.optimized.ll"
$MarkDirtyInputLl = Join-Path $TestDir "mark_dirty_writes.ll"
$MarkDirtyOutputLl = Join-Path $TestDir "mark_dirty_writes.optimized.ll"

Write-Host "Compiling C -> LLVM IR..."
& $Clang -O0 -S -emit-llvm $InputC -o $InputLl

Write-Host "Running compiler_new passes..."
& $MyLlvmOpt `
  --addr-dep-pass `
  --disagg-alloc-pass `
  --disagg-free-pass `
  --mark-dirty-pass `
  --local-addr-pass `
  --addr-dep-rel-pass `
  $InputLl -o $OutputLl

Write-Host "Checking transformed hooks..."
$text = Get-Content $OutputLl -Raw
foreach ($sym in @("disaggAlloc", "disaggFree", "getLocalAddr", "markDirty", "addAddrDep", "updateAddrDep")) {
  if ($text -notmatch $sym) {
    throw "Missing expected hook: $sym"
  }
}
if ($text -match "relAddrDep") {
  throw "Found stale relAddrDep hook"
}
if ($text -notmatch "call ptr @updateAddrDep") {
  throw "Expected updateAddrDep to return the replacement store value"
}

Write-Host "Checking markDirty write coverage..."
& $MyLlvmOpt --mark-dirty-pass $MarkDirtyInputLl -o $MarkDirtyOutputLl
$markDirtyText = Get-Content $MarkDirtyOutputLl -Raw
$markDirtyCalls = ([regex]::Matches($markDirtyText, "call void @markDirty")).Count
if ($markDirtyCalls -ne 12) {
  throw "Expected 12 markDirty calls for write ops, found $markDirtyCalls"
}

Write-Host "Smoke test passed."
