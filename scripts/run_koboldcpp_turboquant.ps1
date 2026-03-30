param(
    [string]$KoboldCppExe = "",
    [switch]$TurboQuantOff,
    [switch]$SO8Off,
    [switch]$TrialityOff,
    [double]$TrialityMix = 0.5,
    [int]$RotationSeed = 0,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$KoboldArgs
)

$ErrorActionPreference = "Stop"

function Resolve-KoboldCppExe {
    param([string]$ExplicitPath)

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        if (Test-Path $ExplicitPath) {
            return (Resolve-Path $ExplicitPath).Path
        }
        throw "KoboldCpp executable not found: $ExplicitPath"
    }

    $candidates = @(
        "$env:USERPROFILE\Downloads\koboldcpp.exe",
        "$env:USERPROFILE\Desktop\koboldcpp.exe",
        "$env:LOCALAPPDATA\Programs\KoboldCpp\koboldcpp.exe",
        "C:\tools\koboldcpp\koboldcpp.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "KoboldCpp executable not found. Use -KoboldCppExe <path-to-koboldcpp.exe>."
}

$exe = Resolve-KoboldCppExe -ExplicitPath $KoboldCppExe

$env:LLAMA_TURBOQUANT = if ($TurboQuantOff) { "0" } else { "1" }
$env:LLAMA_TURBOQUANT_SO8 = if ($SO8Off) { "0" } else { "1" }
$env:LLAMA_TURBOQUANT_TRIALITY = if ($TrialityOff) { "0" } else { "1" }
$env:LLAMA_TURBOQUANT_TRIALITY_MIX = ([Math]::Max(0.0, [Math]::Min(1.0, $TrialityMix))).ToString("0.###")
$env:LLAMA_TURBOQUANT_ROTATION_SEED = [string]$RotationSeed

Write-Host "Launching KoboldCpp with TurboQuant env:"
Write-Host "  LLAMA_TURBOQUANT=$($env:LLAMA_TURBOQUANT)"
Write-Host "  LLAMA_TURBOQUANT_SO8=$($env:LLAMA_TURBOQUANT_SO8)"
Write-Host "  LLAMA_TURBOQUANT_TRIALITY=$($env:LLAMA_TURBOQUANT_TRIALITY)"
Write-Host "  LLAMA_TURBOQUANT_TRIALITY_MIX=$($env:LLAMA_TURBOQUANT_TRIALITY_MIX)"
Write-Host "  LLAMA_TURBOQUANT_ROTATION_SEED=$($env:LLAMA_TURBOQUANT_ROTATION_SEED)"
Write-Host "  EXE=$exe"

& $exe @KoboldArgs
$exitCode = $LASTEXITCODE
if ($null -eq $exitCode) {
    $exitCode = 0
}
exit $exitCode
