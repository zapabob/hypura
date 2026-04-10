# Stop cargo/rustc before building to avoid target/ file locks (required workflow).
# Run from repo root: .\scripts\stop-cargo.ps1

$procs = Get-Process -Name cargo, rustc -ErrorAction SilentlyContinue
if ($procs) {
    $procs | Stop-Process -Force
    Write-Host "Stopped: $($procs.Name -join ', ')"
} else {
    Write-Host "No cargo/rustc processes found."
}
