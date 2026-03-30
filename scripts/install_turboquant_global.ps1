param(
    [string]$BuildBin = "C:\Users\downl\Desktop\hypura-main\hypura-main\vendor\llama.cpp\build-turboquant-cpu\bin\Release",
    [string]$InstallRoot = "$env:LOCALAPPDATA\Programs\llama-turboquant",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$required = @(
    "llama-turboquant.exe",
    "llama.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu.dll"
)

foreach ($name in $required) {
    $path = Join-Path $BuildBin $name
    if (-not (Test-Path $path)) {
        throw "Missing required binary: $path"
    }
}

$binDir = Join-Path $InstallRoot "bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

foreach ($name in $required) {
    $src = Join-Path $BuildBin $name
    $dst = Join-Path $binDir $name
    Copy-Item -Path $src -Destination $dst -Force
}

$shimPath = Join-Path $binDir "tqllama.cmd"
$shimBody = "@echo off`r`n`"%~dp0llama-turboquant.exe`" %*`r`n"
Set-Content -Path $shimPath -Value $shimBody -Encoding ASCII

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$escapedBin = [Regex]::Escape($binDir)
if ($userPath -notmatch "(^|;)$escapedBin(;|$)") {
    $newUserPath = if ([string]::IsNullOrWhiteSpace($userPath)) { $binDir } else { "$userPath;$binDir" }
    [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
    Write-Host "Added to User PATH: $binDir"
} else {
    Write-Host "User PATH already contains: $binDir"
}

Write-Host "Installed binaries to: $binDir"
Write-Host "Use from new shell: llama-turboquant --help"
Write-Host "Shortcut command: tqllama --help"
