# Stops all cargo.exe processes (helps when Cargo blocks on "file lock on artifact directory").
# May interrupt rust-analyzer / other terminals using cargo — use only when stuck.
#
# Usage (repo root):
#   .\scripts\stop-cargo-builds.ps1

$ErrorActionPreference = "Stop"

$procs = @(Get-Process -Name cargo -ErrorAction SilentlyContinue)
if ($procs.Count -eq 0) {
    Write-Host "No cargo.exe processes found."
    exit 0
}
foreach ($p in $procs) {
    Write-Host "Stopping cargo.exe PID $($p.Id)"
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 500
Write-Host "Done. Stopped $($procs.Count) process(es)."
