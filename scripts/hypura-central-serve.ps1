<#
.SYNOPSIS
  RTX30 安定版 dist の hypura.exe で GGUF を Ollama 互換 API の中枢（127.0.0.1:8080）として起動する。

.DESCRIPTION
  環境変数で上書き可能。既定は Qwen3.5 27B Q6_K と同梱 README の推奨に合わせた --context を使用。

  デスクトップ用・2048→8192 状態管理・ショートカット更新は hypura-central-smart.ps1 と Configure-HypuraCentralShortcut.ps1 を参照。

.EXAMPLE
  .\scripts\hypura-central-serve.ps1

.EXAMPLE
  $env:HYPURA_CONTEXT = 4096; .\scripts\hypura-central-serve.ps1
#>
[CmdletBinding()]
param(
    [string] $HypuraExe = $env:HYPURA_EXE,
    [string] $ModelPath = $env:HYPURA_MODEL,
    [string] $Host = $(if ($env:HYPURA_HOST) { $env:HYPURA_HOST } else { "127.0.0.1" }),
    [int] $Port = $(if ($env:HYPURA_PORT) { [int]$env:HYPURA_PORT } else { 8080 }),
    [int] $Context = $(if ($env:HYPURA_CONTEXT) { [int]$env:HYPURA_CONTEXT } else { 2048 })
)

$ErrorActionPreference = "Stop"

# 本スクリプトは <repo>\scripts\ に置く。リポジトリルートはその1つ上。
$repoRoot = Split-Path -Parent $PSScriptRoot
$defaultExe = Join-Path $repoRoot "dist\hypura-rtx30-windows-stable-2026-03-24\hypura.exe"
$defaultModel = "F:\Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf"

if (-not $HypuraExe) { $HypuraExe = $defaultExe }
if (-not $ModelPath) { $ModelPath = $defaultModel }

if (-not (Test-Path -LiteralPath $HypuraExe)) {
    Write-Error "hypura.exe が見つかりません: $HypuraExe （HYPURA_EXE で指定）"
}
if (-not (Test-Path -LiteralPath $ModelPath)) {
    Write-Error "GGUF が見つかりません: $ModelPath （HYPURA_MODEL で指定）"
}

Write-Host "[hypura-central] exe   = $HypuraExe"
Write-Host "[hypura-central] model = $ModelPath"
Write-Host "[hypura-central] bind  = http://${Host}:${Port}/  (Ollama-compatible)"
Write-Host "[hypura-central] context = $Context"
Write-Host ""

& $HypuraExe serve $ModelPath --host $Host --port $Port --context $Context
