<#
.SYNOPSIS
  中枢 Hypura 起動（状態ファイルで context を保持）。2048 安定後に 8192 + 配置エンジン由来の KV ウィンドウ／コンパクトを使う。

.DESCRIPTION
  %LOCALAPPDATA%\Hypura\central-state.json に context を保存。
  - 既定: 2048（初回・Reset）
  - PromoteTo8192 または SmokeAndPromote 成功後: 8192
  Hypura はプレースメントで hot/warm KV を切り、KvCacheManager により長文脈時に自動コンパクト（serve の n_ctx=8192 時も同じ推論パス）。

.PARAMETER PromoteTo8192
  次回以降の起動を 8192 に固定（手動昇格）。

.PARAMETER SmokeAndPromote
  2048 で一時起動→ヘルス＋短い generate が成功したらプロセスを止め、状態を 8192 に書き換え（次回から 8192）。

.PARAMETER ResetTo2048
  状態を 2048 に戻す。
#>
[CmdletBinding(DefaultParameterSetName = "Run")]
param(
    [Parameter(ParameterSetName = "Run")]
    [string] $HypuraExe = $env:HYPURA_EXE,

    [Parameter(ParameterSetName = "Run")]
    [string] $ModelPath = $env:HYPURA_MODEL,

    [Parameter(ParameterSetName = "Run")]
    [string] $BindHost = $(if ($env:HYPURA_HOST) { $env:HYPURA_HOST } else { "127.0.0.1" }),

    [Parameter(ParameterSetName = "Run")]
    [int] $Port = $(if ($env:HYPURA_PORT) { [int]$env:HYPURA_PORT } else { 8080 }),

    [Parameter(ParameterSetName = "Run")]
    [int] $ContextOverride = 0,

    [Parameter(ParameterSetName = "Promote", Mandatory = $true)]
    [switch] $PromoteTo8192,

    [Parameter(ParameterSetName = "Smoke", Mandatory = $true)]
    [switch] $SmokeAndPromote,

    [Parameter(ParameterSetName = "Reset", Mandatory = $true)]
    [switch] $ResetTo2048,

    [Parameter(ParameterSetName = "Info", Mandatory = $true)]
    [switch] $ShowState
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$defaultExe = Join-Path $repoRoot "dist\hypura-rtx30-windows-stable-2026-03-24\hypura.exe"
$defaultModel = "C:\Users\downl\Downloads\Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

$stateDir = Join-Path $env:LOCALAPPDATA "Hypura"
$statePath = Join-Path $stateDir "central-state.json"

function Ensure-Dir {
    if (-not (Test-Path -LiteralPath $stateDir)) {
        New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
    }
}

function Read-State {
    Ensure-Dir
    if (-not (Test-Path -LiteralPath $statePath)) {
        return @{
            context = 16384
            tier    = "safe"
            note    = "default 16384 for Q4_K_M profile"
        }
    }
    $raw = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8
    $j = $raw | ConvertFrom-Json
    $ctx = [int]$j.context
    if ($ctx -ne 2048 -and $ctx -ne 8192 -and $ctx -ne 16384) { $ctx = 16384 }
    return @{
        context = $ctx
        tier    = [string]$j.tier
        note    = [string]$j.note
    }
}

function Write-State {
    param([int]$Context, [string]$Tier)
    Ensure-Dir
    $obj = [ordered]@{
        context          = $Context
        tier             = $Tier
        updated_utc      = (Get-Date).ToUniversalTime().ToString("o")
        kv_compact_note  = "8192: placement splits hot/warm KV; KvCacheManager compacts past hot window (see cache/kv_cache.rs)"
    }
    $obj | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statePath -Encoding UTF8
}

if ($PromoteTo8192) {
    Write-State -Context 8192 -Tier "full"
    Write-Host '[hypura-central-smart] Next launches: --context 8192 (KV window + auto compact)'
    exit 0
}

if ($ResetTo2048) {
    Write-State -Context 2048 -Tier "safe"
    Write-Host '[hypura-central-smart] Next launches: --context 2048'
    exit 0
}

if ($ShowState) {
    $s = Read-State
    Write-Host ($s | ConvertTo-Json -Depth 4)
    if (Test-Path -LiteralPath $statePath) { Write-Host ('File: {0}' -f $statePath) }
    exit 0
}

if (-not $HypuraExe) { $HypuraExe = $defaultExe }
if (-not $ModelPath) { $ModelPath = $defaultModel }

if (-not (Test-Path -LiteralPath $HypuraExe)) {
    Write-Error "hypura.exe not found: $HypuraExe"
}
if (-not (Test-Path -LiteralPath $ModelPath)) {
    Write-Error "GGUF not found: $ModelPath"
}

if ($SmokeAndPromote) {
    $py = Join-Path $repoRoot "scripts\hypura_promote_smoke.py"
    if (-not (Test-Path -LiteralPath $py)) {
        Write-Error "script not found: $py"
    }
    Write-Host '[hypura-central-smart] SmokeAndPromote: py -3 temp serve 2048, smoke, promote state to 8192'
    & py -3 $py --exe $HypuraExe --model $ModelPath --host $BindHost --port $Port --state-path $statePath
    exit $LASTEXITCODE
}

# --- 通常起動 ---
$st = Read-State
$ctx = $st.context
if ($ContextOverride -gt 0) {
    $ctx = $ContextOverride
}
if ($env:HYPURA_CONTEXT) {
    $ctx = [int]$env:HYPURA_CONTEXT
}

Write-Host ('[hypura-central-smart] state file: {0}' -f $statePath)
Write-Host ('[hypura-central-smart] exe     = {0}' -f $HypuraExe)
Write-Host ('[hypura-central-smart] model   = {0}' -f $ModelPath)
Write-Host ('[hypura-central-smart] bind    = http://{0}:{1}/' -f $BindHost, $Port)
Write-Host ('[hypura-central-smart] context = {0}  (tier={1})' -f $ctx, $st.tier)
if ($ctx -eq 8192) {
    Write-Host '[hypura-central-smart] KV: long context uses hot window + compaction when warm > 0 (Hypura placement)'
}
Write-Host ""

& $HypuraExe serve $ModelPath --host $BindHost --port $Port --context $ctx
