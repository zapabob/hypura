param(
    [ValidateSet("check", "full")]
    [string]$Mode = "check"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildRoot = "F:\hypura-build"
$targetDir = Join-Path $buildRoot "target"
$cargoHome = Join-Path $buildRoot "cargo-home"
$rustupHome = Join-Path $buildRoot "rustup-home"
$logDir = Join-Path $repoRoot "logs"
$logPath = Join-Path $logDir "build-check.log"
$summaryPath = Join-Path $logDir "build-check-summary.json"

if (-not (Test-Path "F:\")) {
    throw "F: drive is required for hypura-build-check.ps1"
}

foreach ($path in @($buildRoot, $targetDir, $cargoHome, $logDir)) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

$env:CARGO_TARGET_DIR = $targetDir
$env:CARGO_HOME = $cargoHome
$env:RUSTC_WRAPPER = ""
$env:CARGO_INCREMENTAL = "0"

if ($env:HYPURA_RUSTUP_HOME) {
    New-Item -ItemType Directory -Force -Path $env:HYPURA_RUSTUP_HOME | Out-Null
    $env:RUSTUP_HOME = $env:HYPURA_RUSTUP_HOME
}

$commands = @(
    [pscustomobject]@{
        Exe  = "cargo"
        Args = @("check", "--workspace", "--message-format", "short")
    }
)

if ($Mode -eq "full") {
    $commands += [pscustomobject]@{
        Exe  = "cargo"
        Args = @("test", "--workspace", "--lib", "--message-format", "short")
    }
}

if (Test-Path $logPath) {
    Remove-Item $logPath -Force
}

$allOutput = New-Object System.Collections.Generic.List[string]
$failedCommand = $null
$exitCode = 0

foreach ($command in $commands) {
    $exe = $command.Exe
    $args = @($command.Args)

    $cmdLine = $exe
    if ($args.Count -gt 0) {
        $cmdLine = "$exe $($args -join ' ')"
    }
    ">>> $cmdLine" | Tee-Object -FilePath $logPath -Append | Out-Null

    $output = & cmd.exe /c "$cmdLine 2>&1"
    $output | Tee-Object -FilePath $logPath -Append | Out-Null
    foreach ($line in $output) {
        $allOutput.Add([string]$line)
    }

    if ($LASTEXITCODE -ne 0) {
        $failedCommand = $cmdLine
        $exitCode = $LASTEXITCODE
        break
    }
}

$warningLines = @($allOutput | Where-Object { $_ -match 'warning:' })
$errorLines = @($allOutput | Where-Object { $_ -match 'error:' })
$firstError = if ($errorLines.Count -gt 0) { $errorLines[0] } else { $null }
$firstWarning = if ($warningLines.Count -gt 0) { $warningLines[0] } else { $null }

$summary = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    mode = $Mode
    success = ($failedCommand -eq $null -and $warningLines.Count -eq 0)
    build_root = $buildRoot
    cargo_target_dir = $targetDir
    cargo_home = $cargoHome
    rustup_home = $(if ($env:RUSTUP_HOME) { $env:RUSTUP_HOME } else { $null })
    commands = @($commands | ForEach-Object {
        if ($_.Args.Count -gt 0) {
            "$($_.Exe) $($_.Args -join ' ')"
        } else {
            $_.Exe
        }
    })
    warning_count = $warningLines.Count
    error_count = $errorLines.Count
    failed_command = $failedCommand
    exit_code = $exitCode
    first_warning = $firstWarning
    first_error = $firstError
    log_path = $logPath
}

$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $summaryPath
Write-Host "Wrote summary to $summaryPath"

if (-not $summary.success) {
    exit 1
}
