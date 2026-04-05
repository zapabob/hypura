# F: first, then H: for CARGO_TARGET_DIR; incremental build; copy Hypura.exe + hypura.exe into repo target\release (and optional install dirs).
# Requires ~15 GiB free on chosen drive. Unsets RUSTC_WRAPPER (sccache + incremental conflict).
# Default: RUSTFLAGS=-D warnings (zero compiler warnings). Use -AllowWarnings to opt out.
#
# Usage (from repo root):
#   .\scripts\build-hypura-hf.ps1
#   .\scripts\build-hypura-hf.ps1 -Debug
#   .\scripts\build-hypura-hf.ps1 -NoDefaultFeatures
#   .\scripts\build-hypura-hf.ps1 -AlsoCopyTo "F:\Hypura"
#   .\scripts\build-hypura-hf.ps1 -InstallTo "F:\Tools\Hypura"
#   .\scripts\build-hypura-hf.ps1 -NoKill
#   .\scripts\build-hypura-hf.ps1 -AllowWarnings
#   .\scripts\build-hypura-hf.ps1 -PasteCommands
#   .\scripts\build-hypura-hf.ps1 -Clipboard
#   .\scripts\build-hypura-hf.ps1 -CargoOverride "D:\tools\cargo.exe"
#   .\scripts\build-hypura-hf.ps1 -StopOtherCargo
# Env: HYPURA_CARGO_EXE or HYPURA_CARGO

param(
    [switch] $Debug,
    [switch] $NoDefaultFeatures,
    [switch] $NoCopyToRepo,
    [string[]] $InstallTo,
    [string] $AlsoCopyTo,
    [int] $MinFreeGiB = 15,
    [switch] $AllowWarnings,
    [switch] $NoKill,
    [switch] $PasteCommands,
    [switch] $Clipboard,
    [string] $CargoOverride = "",
    [switch] $StopOtherCargo
)

$ErrorActionPreference = "Stop"

function Get-SingleQuotedLiteral([string]$PathValue) {
    if ($null -eq $PathValue) { return "''" }
    return "'" + ($PathValue.Replace("'", "''")) + "'"
}

function Get-CargoExe {
    param([string]$Override = "")
    function Resolve-ExistingExe([string]$PathValue) {
        if (-not $PathValue) { return $null }
        if (-not (Test-Path -LiteralPath $PathValue)) { return $null }
        return (Resolve-Path -LiteralPath $PathValue).Path
    }
    $tryFirst = @(
        $Override,
        $env:HYPURA_CARGO_EXE,
        $env:HYPURA_CARGO
    )
    foreach ($t in $tryFirst) {
        $r = Resolve-ExistingExe $t
        if ($r) { return $r }
    }
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:CARGO_HOME) {
        $candidates.Add((Join-Path $env:CARGO_HOME "bin\cargo.exe"))
    }
    $candidates.Add((Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"))
    $scoopShim = Join-Path $env:USERPROFILE "scoop\shims\cargo.exe"
    $candidates.Add($scoopShim)
    foreach ($p in $candidates) {
        $r = Resolve-ExistingExe $p
        if ($r) { return $r }
    }
    $toolchainRoots = New-Object System.Collections.Generic.List[string]
    if ($env:RUSTUP_HOME) {
        $toolchainRoots.Add((Join-Path $env:RUSTUP_HOME "toolchains"))
    }
    $toolchainRoots.Add((Join-Path $env:USERPROFILE ".rustup\toolchains"))
    foreach ($toolchains in $toolchainRoots) {
        if (-not (Test-Path $toolchains)) { continue }
        foreach ($tc in Get-ChildItem -Path $toolchains -Directory -ErrorAction SilentlyContinue) {
            $cand = Join-Path $tc.FullName "bin\cargo.exe"
            $r = Resolve-ExistingExe $cand
            if ($r) { return $r }
        }
    }
    $cmd = Get-Command cargo -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }
    $whereExe = Join-Path $env:SystemRoot "System32\where.exe"
    if (Test-Path -LiteralPath $whereExe) {
        $lines = & $whereExe cargo 2>$null
        if ($lines) {
            foreach ($line in $lines) {
                $t = $line.Trim()
                $r = Resolve-ExistingExe $t
                if ($r) { return $r }
            }
        }
    }
    $cargoErr = [string]::Join(
        [Environment]::NewLine,
        [string[]]@(
            'cargo.exe not found.',
            '',
            '1. Install Rust from https://rustup.rs/ and restart the terminal.',
            '2. Or set env HYPURA_CARGO_EXE to the full path of cargo.exe.',
            '3. Or run: .\scripts\build-hypura-hf.ps1 -CargoOverride C:\path\to\cargo.exe',
            '4. User and Machine PATH are merged in this script; if cargo is still missing, Rust is not installed.',
            '5. Code Runner often lacks PATH; prefer VS Code Integrated Terminal.'
        )
    )
    throw $cargoErr
}

function Stop-HypuraProcesses {
    foreach ($name in @("Hypura", "hypura")) {
        Get-Process -Name $name -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Host "Stopping process: $($_.ProcessName) (PID $($_.Id))"
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Milliseconds 500
}

function Stop-OtherCargoProcesses {
    $procs = @(Get-Process -Name cargo -ErrorAction SilentlyContinue)
    foreach ($p in $procs) {
        Write-Host "Stopping cargo.exe (PID $($p.Id)) for artifact lock"
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
    if ($procs.Count -gt 0) {
        Start-Sleep -Milliseconds 800
    }
}

function Show-CargoLockHint {
    $n = @(Get-Process -Name cargo -ErrorAction SilentlyContinue).Count
    if ($n -gt 0) {
        Write-Warning "cargo.exe already running ($n process(es)). If build blocks on file lock, wait, close other terminals, run .\scripts\stop-cargo-builds.ps1, or use -StopOtherCargo."
    }
}

function Get-DriveFreeGiB([string]$RootPath) {
    if (-not ($RootPath -match '^([A-Za-z]):')) { return 0 }
    $letter = $Matches[1].ToUpperInvariant()
    try {
        $vol = Get-Volume -DriveLetter $letter -ErrorAction Stop
        return [math]::Round($vol.SizeRemaining / 1GB, 2)
    } catch {
        return 0
    }
}

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$pref = New-Object System.Collections.Generic.List[string]
if ($userPath) { $pref.Add($userPath) }
if ($machinePath) { $pref.Add($machinePath) }
if ($pref.Count -gt 0) {
    $env:Path = "$([string]::Join([IO.Path]::PathSeparator, $pref));$env:Path"
}
$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if (Test-Path $cargoBin) {
    $env:Path = "$cargoBin;$env:Path"
}
if ($env:CARGO_HOME) {
    $cargoHomeBin = Join-Path $env:CARGO_HOME "bin"
    if (Test-Path $cargoHomeBin) {
        $env:Path = "$cargoHomeBin;$env:Path"
    }
}

$candidateDirs = @(
    @{ Path = "F:\hypura-cargo-target"; Drive = 'F:\' },
    @{ Path = "H:\hypura-cargo-target"; Drive = 'H:\' }
)

$targetDir = $null
foreach ($c in $candidateDirs) {
    if (-not (Test-Path $c.Drive)) { continue }
    $free = Get-DriveFreeGiB $c.Path
    if ($free -lt $MinFreeGiB) {
        Write-Warning "$($c.Drive) free space is ${free} GiB (need >= ${MinFreeGiB} GiB). Skipping."
        continue
    }
    $targetDir = $c.Path
    Write-Host "Using CARGO_TARGET_DIR=$targetDir (free ~${free} GiB on drive)"
    break
}

if (-not $targetDir) {
    throw "No usable F: or H: (missing drive or free space < ${MinFreeGiB} GiB). Free disk and retry."
}

if (-not $AllowWarnings) {
    $env:RUSTFLAGS = "-D warnings"
    Write-Host "RUSTFLAGS=-D warnings (use -AllowWarnings to disable)"
} else {
    Remove-Item Env:\RUSTFLAGS -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
$env:CARGO_TARGET_DIR = $targetDir
$env:CARGO_INCREMENTAL = "1"
Remove-Item Env:\RUSTC_WRAPPER -ErrorAction SilentlyContinue
$env:TMP = Join-Path $env:LOCALAPPDATA "Temp"
$env:TEMP = $env:TMP

$cargoArgs = @("build", "-p", "hypura")
if (-not $Debug) { $cargoArgs += "--release" }
if ($NoDefaultFeatures) {
    $cargoArgs += "--no-default-features"
} else {
    $cargoArgs += "--features", "kobold-gui"
}

$cargoExe = Get-CargoExe -Override $CargoOverride
Write-Host "Using cargo: $cargoExe"
if ($StopOtherCargo) {
    Stop-OtherCargoProcesses
} else {
    Show-CargoLockHint
}
Write-Host "cargo $($cargoArgs -join ' ')"
& $cargoExe @cargoArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$profile = if ($Debug) { "debug" } else { "release" }
$built = Join-Path (Join-Path $targetDir $profile) "Hypura.exe"
if (-not (Test-Path -LiteralPath $built)) {
    throw "Built EXE not found: $built"
}

Write-Host "Built: $built"

$copyDestinations = New-Object System.Collections.Generic.List[string]
if (-not $NoCopyToRepo) {
    $copyDestinations.Add((Join-Path $root "target\$profile"))
}
if ($AlsoCopyTo) { $copyDestinations.Add($AlsoCopyTo) }
if ($InstallTo) {
    foreach ($d in $InstallTo) {
        if ($d) { $copyDestinations.Add($d) }
    }
}

if ($copyDestinations.Count -gt 0 -and -not $NoKill) {
    Stop-HypuraProcesses
}

function Copy-HypuraBinaries([string]$destDir) {
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    $hypura = Join-Path $destDir "Hypura.exe"
    $legacy = Join-Path $destDir "hypura.exe"
    Copy-Item -LiteralPath $built -Destination $hypura -Force
    Copy-Item -LiteralPath $built -Destination $legacy -Force -ErrorAction SilentlyContinue
    Write-Host "Installed (overwrite): $hypura"
}

foreach ($destDir in $copyDestinations) {
    if ($destDir) { Copy-HypuraBinaries $destDir }
}

if ($PasteCommands -or $Clipboard) {
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# Hypura binary overwrite (paste into PowerShell)")
    $lines.Add("`$src = $(Get-SingleQuotedLiteral $built)")
    foreach ($destDir in $copyDestinations) {
        if (-not $destDir) { continue }
        $hypPath = Join-Path $destDir "Hypura.exe"
        $legPath = Join-Path $destDir "hypura.exe"
        $lines.Add("New-Item -ItemType Directory -Force -Path $(Get-SingleQuotedLiteral $destDir) | Out-Null")
        $lines.Add("Copy-Item -LiteralPath `$src -Destination $(Get-SingleQuotedLiteral $hypPath) -Force")
        $lines.Add("Copy-Item -LiteralPath `$src -Destination $(Get-SingleQuotedLiteral $legPath) -Force -ErrorAction SilentlyContinue")
    }
    $block = ($lines -join "`n")
    Write-Host ""
    Write-Host "=== コピペ用 (手動で別先へ上書きコピー) ===" -ForegroundColor Cyan
    Write-Host $block
    if ($Clipboard) {
        Set-Clipboard -Value $block
        Write-Host "(clipboard にコピー済み)" -ForegroundColor Green
    }
}
