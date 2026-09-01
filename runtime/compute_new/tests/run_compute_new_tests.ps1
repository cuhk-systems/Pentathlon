Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [string]$RepoRoot = "",
    [string]$BuildDir = "",
    [string]$BuildType = "RelWithDebInfo",
    [string]$MemoryAddr = "",
    [int]$Jobs = 8,
    [switch]$RunBatchRead
)

function Resolve-DefaultRepoRoot {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    return (Resolve-Path (Join-Path $scriptDir "..\..\..")).Path
}

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Resolve-DefaultRepoRoot
} else {
    $RepoRoot = (Resolve-Path $RepoRoot).Path
}

if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $RepoRoot "runtime\build_compute_new"
}

if (-not [string]::IsNullOrWhiteSpace($MemoryAddr)) {
    $env:MEMORY_ADDR = $MemoryAddr
}

if ([string]::IsNullOrWhiteSpace($env:MEMORY_ADDR)) {
    throw "MEMORY_ADDR is not set. Pass -MemoryAddr <ip-or-host> or set `$env:MEMORY_ADDR first."
}

$runtimeDir = Join-Path $RepoRoot "runtime"

Write-Host "RepoRoot: $RepoRoot"
Write-Host "Runtime dir: $runtimeDir"
Write-Host "Build dir: $BuildDir"
Write-Host "Build type: $BuildType"
Write-Host "MEMORY_ADDR: $env:MEMORY_ADDR"

Push-Location $runtimeDir
try {
    & cmake -S . -B $BuildDir -G Ninja -DPENTATHLON_USE_COMPUTE_NEW=ON -DCMAKE_BUILD_TYPE=$BuildType
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed with exit code $LASTEXITCODE"
    }

    & cmake --build $BuildDir -j $Jobs
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }

    $tests = @(
        "dm_compiler_rt_compute_tests",
        "dm_compiler_rt_compute_tests_c"
    )
    if ($RunBatchRead) {
        $tests += "test-batch-read"
    }

    foreach ($testName in $tests) {
        $exePath = Join-Path $BuildDir ($testName + ".exe")
        if (-not (Test-Path $exePath)) {
            throw "Test binary not found: $exePath"
        }
        Write-Host "`n== Running $testName =="
        & $exePath
        if ($LASTEXITCODE -ne 0) {
            throw "$testName failed with exit code $LASTEXITCODE"
        }
    }

    Write-Host "`nAll selected compute_new tests passed."
}
finally {
    Pop-Location
}
