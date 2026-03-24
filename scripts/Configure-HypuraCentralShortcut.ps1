<#
.SYNOPSIS
  デスクトップの既存ショートカットを検索し、Hypura 中枢ランチャー（hypura-central-smart.ps1）を指すよう更新する。無ければ新規作成する。

.DESCRIPTION
  マッチ条件（いずれか）:
  - 名前に Hypura / hypura / central が含まれる .lnk
  - リンク先または引数に hypura.exe / hypura-central-serve / hypura-central-smart が含まれる

.EXAMPLE
  .\scripts\Configure-HypuraCentralShortcut.ps1
#>
[CmdletBinding()]
param(
    [string] $ShortcutName = "Hypura 中枢 (Ollama API).lnk"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $repoRoot "scripts\hypura-central-smart.ps1"
if (-not (Test-Path -LiteralPath $launcher)) {
    Write-Error "ランチャーが見つかりません: $launcher"
}

$desktop = [Environment]::GetFolderPath("Desktop")
$wsh = New-Object -ComObject WScript.Shell

$psExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$argsLine = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Normal -File `"$launcher`""

function Test-LnkMatch([string]$path) {
    $n = [System.IO.Path]::GetFileName($path)
    if ($n -match 'hypura|Hypura|central|中枢') { return $true }
    $s = $wsh.CreateShortcut($path)
    $t = "$($s.TargetPath) $($s.Arguments)"
    if ($t -match 'hypura|Hypura|central-serve|central-smart') { return $true }
    return $false
}

$updated = @()
Get-ChildItem -Path $desktop -Filter *.lnk -File -ErrorAction SilentlyContinue | ForEach-Object {
    if (Test-LnkMatch $_.FullName) {
        $sc = $wsh.CreateShortcut($_.FullName)
        $sc.TargetPath = $psExe
        $sc.Arguments = $argsLine
        $sc.WorkingDirectory = $repoRoot
        $sc.Description = "Hypura 中枢: state %LOCALAPPDATA%\Hypura\central-state.json (2048→8192 + KV compact)"
        $sc.Save()
        $updated += $_.Name
    }
}

$newPath = Join-Path $desktop $ShortcutName
$scNew = $wsh.CreateShortcut($newPath)
$scNew.TargetPath = $psExe
$scNew.Arguments = $argsLine
$scNew.WorkingDirectory = $repoRoot
$scNew.Description = "Hypura 中枢: state %LOCALAPPDATA%\Hypura\central-state.json"
$scNew.Save()
Write-Host "[shortcut] primary (always write): $newPath"

if ($updated.Count -gt 0) {
    Write-Host "[shortcut] updated: $($updated -join ', ')"
} else {
    Write-Host "[shortcut] no matching .lnk on Desktop (created new if missing)"
}

Write-Host "[shortcut] launcher: $launcher"
Write-Host "[shortcut] first run uses context 2048; then: .\scripts\hypura-central-smart.ps1 -SmokeAndPromote  OR  -PromoteTo8192"
