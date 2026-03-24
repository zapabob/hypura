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
    [string] $BindHost = $(if ($env:HYPURA_HOST) { $env:HYPURA_HOST } else { "127.0.0.1" }),
    [int] $Port = $(if ($env:HYPURA_PORT) { [int]$env:HYPURA_PORT } else { 8080 }),
    [int] $Context = $(if ($env:HYPURA_CONTEXT) { [int]$env:HYPURA_CONTEXT } else { 2048 })
)

$ErrorActionPreference = "Stop"

# 本スクリプトは <repo>\scripts\ に置く。リポジトリルートはその1つ上。
$repoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-HypuraExe {
    if ($HypuraExe -and (Test-Path -LiteralPath $HypuraExe)) {
        return $HypuraExe
    }

    $releaseExe = Join-Path $repoRoot "target\release\hypura.exe"
    if (Test-Path -LiteralPath $releaseExe) {
        return $releaseExe
    }

    $distExe = Get-ChildItem -Path (Join-Path $repoRoot "dist") -Filter "hypura.exe" -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($distExe) {
        return $distExe.FullName
    }

    return $null
}

function Resolve-ModelPath {
    if ($ModelPath -and (Test-Path -LiteralPath $ModelPath)) {
        return $ModelPath
    }

    $candidates = @(
        (Join-Path $repoRoot "test-models"),
        (Join-Path $repoRoot "models")
    )

    foreach ($dir in $candidates) {
        if (-not (Test-Path -LiteralPath $dir)) { continue }
        $model = Get-ChildItem -Path $dir -Filter "*.gguf" -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($model) {
            return $model.FullName
        }
    }

    return $null
}

$HypuraExe = Resolve-HypuraExe
$ModelPath = Resolve-ModelPath

if (-not (Test-Path -LiteralPath $HypuraExe)) {
    Write-Error "hypura.exe が見つかりません。HYPURA_EXE を設定するか target\release\hypura.exe / dist\**\hypura.exe を配置してください。"
}
if (-not (Test-Path -LiteralPath $ModelPath)) {
    Write-Error "GGUF が見つかりません。HYPURA_MODEL を設定するか test-models/ または models/ 配下に *.gguf を置いてください。"
}

Write-Host "[hypura-central] exe   = $HypuraExe"
Write-Host "[hypura-central] model = $ModelPath"
Write-Host "[hypura-central] bind  = http://${BindHost}:${Port}/  (Ollama-compatible)"
Write-Host "[hypura-central] context = $Context"
Write-Host ""

& $HypuraExe serve $ModelPath --host $BindHost --port $Port --context $Context
