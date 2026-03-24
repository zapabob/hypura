#requires -Version 5.1
<#
  Hypura + OpenClaw 整合チェック（秘密情報は出さない）
  exit 0: Hypura 応答あり、かつ primary と /api/tags が比較可能で一致（または hypura primary でない等で比較スキップ）
  exit 1: 未応答、または名前不一致
#>
param(
    [int] $Port = 8080,
    [string] $HypuraHost = '127.0.0.1'
)

$ErrorActionPreference = 'Continue'
$runId = "run_{0}" -f ([Guid]::NewGuid().ToString('N').Substring(0, 8))

# --- Config path (OpenClaw) ---
$configPath = $null
if ($env:OPENCLAW_CONFIG_PATH -and $env:OPENCLAW_CONFIG_PATH.Trim().Length -gt 0) {
    $configPath = $env:OPENCLAW_CONFIG_PATH.Trim()
}
elseif ($env:OPENCLAW_STATE_DIR -and $env:OPENCLAW_STATE_DIR.Trim().Length -gt 0) {
    $configPath = Join-Path $env:OPENCLAW_STATE_DIR.Trim() 'openclaw.json'
}
else {
    $configPath = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.openclaw\openclaw.json'
}

$primaryModel = $null
$hypuraBaseUrl = $null
if (Test-Path -LiteralPath $configPath) {
    try {
        $raw = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8
        $cfg = $raw | ConvertFrom-Json
        $primaryModel = $cfg.agents.defaults.model.primary
        $hypuraBaseUrl = $cfg.models.providers.hypura.baseUrl
    }
    catch {
        Write-Warning "Config parse error: $($_.Exception.Message)"
    }
}

# --- Hypura HTTP ---
$base = "http://${HypuraHost}:${Port}"
$healthOk = $false
$tagsName = $null
try {
    $r = Invoke-WebRequest -Uri ($base + '/') -UseBasicParsing -TimeoutSec 5
    $healthOk = ($r.StatusCode -eq 200)
}
catch { }

if ($healthOk) {
    try {
        $tags = Invoke-RestMethod -Uri ($base + '/api/tags') -TimeoutSec 30
        $first = $tags.models | Select-Object -First 1
        if ($first) { $tagsName = [string]$first.name }
    }
    catch { }
}

# --- baseUrl port vs -Port ---
$portMismatch = $null
if ($hypuraBaseUrl) {
    try {
        $u = [Uri]$hypuraBaseUrl
        $portInUrl = $u.Port
        $portMismatch = ($portInUrl -ne $Port)
    }
    catch { }
}

$primarySuffix = $null
if ($primaryModel -match '^hypura/(.+)$') { $primarySuffix = $Matches[1] }
$nameMatch = $null
if ($null -ne $tagsName -and $null -ne $primarySuffix) { $nameMatch = ($tagsName -eq $primarySuffix) }

# --- Console output ---
Write-Host ""
Write-Host "=== Hypura / OpenClaw check (run=$runId) ===" -ForegroundColor Cyan
Write-Host "OpenClaw config path: $configPath (exists: $(Test-Path -LiteralPath $configPath))"
Write-Host "  primary:              $primaryModel"
Write-Host "  hypura baseUrl:       $hypuraBaseUrl"
Write-Host "Hypura probe:           $base/"
Write-Host "  health OK:            $healthOk"
if ($healthOk) {
    Write-Host "  /api/tags name:       $tagsName"
}
Write-Host "Port check (baseUrl vs -Port $Port): $(if ($null -eq $portMismatch) { 'n/a' } else { -not $portMismatch })"
Write-Host "primary vs tags match:  $(if ($null -eq $nameMatch) { 'n/a (no tags or non-hypura primary)' } else { $nameMatch })"
Write-Host ""

if (-not $healthOk) {
    $listenCount = -1
    try {
        $listenCount = @(
            Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        ).Count
    }
    catch { $listenCount = -1 }
    Write-Host "  TCP Listen count on port ${Port}: $listenCount (0=no listener; >=1=something bound)" -ForegroundColor DarkGray

    Write-Host "BLOCKER: Nothing responded on $base - start Hypura first (e.g. hypura-central-smart.ps1)." -ForegroundColor Yellow
    exit 1
}

if ($null -eq $nameMatch) {
    Write-Host "OK: Hypura up. Could not compare names (no tags or primary not hypura/...)." -ForegroundColor Green
    exit 0
}

if (-not $nameMatch) {
    Write-Host "FAIL: primary suffix '$primarySuffix' != tags name '$tagsName'" -ForegroundColor Red
    exit 1
}

Write-Host "OK: Config and Hypura tags agree." -ForegroundColor Green
exit 0
