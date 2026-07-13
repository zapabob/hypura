# Hypura release procedure

This procedure produces the Windows CUDA 12.8, RTX 50-series `sm_120` release for Hypura. A release is complete only after source pins, tests, live CLI/API behavior, installers, overwrite installation, GitHub branches, tags, releases, and uploaded checksums have all been verified.

## Release contract

The v1.0.0 source authorities are `zapabob/llama.cpp` and `zapabob/Turboquant-CUDA`. The Hypura release commit must pin tested commits from those repositories. The recursive fullsource archive records Hypura, both direct submodules, and the nested Turboquant llama.cpp commit in `SOURCE-MANIFEST.json`.

The two publication tracks point to the same tested release commit:

- stable branch: `stable/v1.0.0`, tag and release `v1.0.0`
- main branch: `main`, tag and release `main-v1.0.0`

Both releases carry the CLI executable, recursive fullsource archive, fullsource checksum, MSI, NSIS installer, and `SHA256SUMS.txt`.

## 1. Preflight

Run all release work from a clean isolated worktree. Do not release from a worktree that contains unrelated local changes.

```powershell
gh auth status
git status --short --branch
git submodule status --recursive
git diff --check
```

Confirm the version is `1.0.0` in the root Cargo workspace, Desktop Cargo workspace, Tauri configuration, npm package, and both lockfile families.

```powershell
rg -n '0\.15\.0|version = "1\.0\.0"|"version": "1\.0\.0"' `
  Cargo.toml Cargo.lock `
  hypura-desktop/Cargo.toml hypura-desktop/Cargo.lock `
  hypura-desktop/package.json hypura-desktop/package-lock.json `
  hypura-desktop/src-tauri/tauri.conf.json
```

## 2. Tests and source gates

Run Turboquant schema-v2 export, verification, round-trip, and negative tests. Run llama.cpp Triality contract, consensus, storage, telemetry, and CUDA parity tests. Run the Hypura unit and integration suites with a target directory on a drive with adequate capacity.

```powershell
$env:CARGO_TARGET_DIR = 'H:\cargo-targets\hypura-v1-tests'
cargo test --workspace --locked
```

Any skipped GPU test must have a recorded reason. The stable release requires live RTX 50-series CUDA evidence; CPU-only results do not satisfy that gate.

## 3. CUDA 12.8 `sm_120` CLI build

The ambient CUDA installation is not authoritative. Set every CUDA selector explicitly and clean `hypura-sys` before the release build.

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
$env:HYPURA_CUDA = '1'
$env:HYPURA_CUDA_ARCHITECTURES = '120'
$env:CMAKE_CUDA_ARCHITECTURES = '120'
$env:CARGO_BUILD_JOBS = '6'
$env:CMAKE_BUILD_PARALLEL_LEVEL = '6'
$env:LLAMA_BUILD_UI = 'OFF'
$env:LLAMA_BUILD_WEBUI = 'OFF'
$env:CARGO_TARGET_DIR = 'H:\hypura-cargo-target-v1.0.0'

$activeRustBuilds = Get-CimInstance Win32_Process | Where-Object {
  $_.Name -in @('cargo.exe', 'rustc.exe')
}
if ($activeRustBuilds) {
  $activeRustBuilds | Select-Object ProcessId,ParentProcessId,Name,CommandLine
  throw 'Rust builds are active. Identify ownership and wait for or stop only release-owned processes before cleaning.'
}

$nvcc = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
$nvccVersion = (& $nvcc --version | Out-String)
if ($LASTEXITCODE -ne 0 -or $nvccVersion -notmatch 'release 12\.8') {
  throw 'The selected compiler is not CUDA 12.8 nvcc.'
}

cargo clean -p hypura-sys
if ($LASTEXITCODE -ne 0) { throw 'cargo clean failed' }
cargo build --release --locked --bin hypura
if ($LASTEXITCODE -ne 0) { throw 'release CLI build failed' }
```

Verify the executable before it is copied anywhere.

```powershell
$cli = 'H:\hypura-cargo-target-v1.0.0\release\Hypura.exe'
$cliVersion = (& $cli --version | Out-String)
if ($LASTEXITCODE -ne 0 -or $cliVersion -notmatch '\b1\.0\.0\b') {
  throw 'release CLI version verification failed'
}
& $cli --help
if ($LASTEXITCODE -ne 0) { throw 'release CLI help failed' }
& $cli council --help
if ($LASTEXITCODE -ne 0) { throw 'release Council help failed' }
& $cli council .\model.gguf --prompt 'Release smoke test' --max-tokens 8 --cross-score
if ($LASTEXITCODE -ne 0) { throw 'release Council smoke test failed' }
& $cli council .\missing.gguf --prompt 'bad input' --cross-score
if ($LASTEXITCODE -eq 0) { throw 'invalid model input unexpectedly succeeded' }

$dumpbin = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\dumpbin.exe'
$releaseBinaries = @($cli) + @(
  Get-ChildItem -LiteralPath (Split-Path -Parent $cli) -Filter '*.dll' -File
)
$dependencyProof = foreach ($binary in $releaseBinaries) {
  "### $binary"
  & $dumpbin /DEPENDENTS $binary
  if ($LASTEXITCODE -ne 0) { throw "dumpbin failed for $binary" }
}
if (($dependencyProof | Out-String) -match '(?i)\b[A-Za-z0-9_]*64_13\d*\.dll\b') {
  throw 'A CUDA 13 runtime dependency was found.'
}

$cmakeCache = Get-ChildItem -LiteralPath $env:CARGO_TARGET_DIR `
  -Filter CMakeCache.txt -File -Recurse |
  Where-Object { $_.FullName -match 'hypura-sys' } |
  Select-Object -First 1
if (-not $cmakeCache) { throw 'hypura-sys CMakeCache.txt was not found' }
$cacheText = Get-Content -LiteralPath $cmakeCache.FullName -Raw
if ($cacheText -notmatch '(?i)CMAKE_CUDA_COMPILER:FILEPATH=.*CUDA[/\\]v12\.8[/\\]bin[/\\]nvcc\.exe') {
  throw 'CMake did not record the CUDA 12.8 compiler.'
}
if ($cacheText -notmatch '(?m)^CMAKE_CUDA_ARCHITECTURES(?::[^=]+)?=120$') {
  throw 'CMake did not record exact architecture 120.'
}

$cuobjdump = Join-Path $env:CUDA_PATH 'bin\cuobjdump.exe'
$cubinProof = foreach ($binary in $releaseBinaries) {
  $output = & $cuobjdump --list-elf $binary 2>&1
  if ($LASTEXITCODE -eq 0) {
    "### $binary"
    $output
  }
}
$architectures = [regex]::Matches(
  ($cubinProof | Out-String),
  'sm_[0-9]+[a-z]?'
) | ForEach-Object Value | Sort-Object -Unique
if (@($architectures).Count -ne 1 -or $architectures[0] -ne 'sm_120') {
  throw "Expected only sm_120 cubins, found: $($architectures -join ', ')"
}
```

The version must be `1.0.0`, valid generation must complete, invalid input must fail non-zero, nvcc and CMake must both identify CUDA 12.8 with architecture 120, every embedded cubin must be `sm_120`, and no CUDA 13 runtime DLL may appear in the dependency proof.

When `model.gguf` is the documented identity-view wiring fixture rather than a production multi-view model, the valid-generation command must additionally pass `--tq-developer-override --tq-allow-identity-view-fallback`. Those switches are fixture-only and must never be used to claim model-quality improvement.

## 4. HTTP manual QA

Start the verified CLI against the release model, then exercise the Council endpoint and trace retrieval from another terminal.

```powershell
& $cli serve .\model.gguf --host 127.0.0.1 --port 8080
```

For the identity-view wiring fixture, append `--tq-developer-override --tq-allow-identity-view-fallback` to the server command as well.

```powershell
$request = @{
  prompt = 'Release smoke test'
  parallelism = 'auto'
  cross_score = $true
  max_tokens = 8
  stream = $false
} | ConvertTo-Json

$result = Invoke-RestMethod `
  -Method Post `
  -Uri 'http://127.0.0.1:8080/api/extra/triality/council' `
  -ContentType 'application/json' `
  -Body $request

Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8080/api/extra/triality/council/$($result.id)"
```

Also verify one unchanged native endpoint and one KoboldCpp-compatible endpoint to detect regressions outside the new surface.

## 5. Desktop installers

```powershell
Set-Location .\hypura-desktop
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
$env:HYPURA_CUDA = '1'
$env:HYPURA_CUDA_ARCHITECTURES = '120'
$env:CMAKE_CUDA_ARCHITECTURES = '120'
$env:TMP = 'H:\hypura-tmp'
$env:TEMP = 'H:\hypura-tmp'
$env:npm_config_cache = 'H:\npm-cache'
$env:RUSTC_WRAPPER = ''
$env:CARGO_TARGET_DIR = 'H:\hypura-desktop-release-target-v1.0.0'

npm ci
if ($LASTEXITCODE -ne 0) { throw 'Desktop npm dependency installation failed' }
npm run tauri -- build
if ($LASTEXITCODE -ne 0) { throw 'Desktop MSI/NSIS build failed' }
Set-Location ..
```

Record whether the installers are signed. If no signing certificate is available, release notes must state that the MSI and NSIS artifacts are unsigned.

## 6. Recursive fullsource

Create the archive from the tested release commit, not from uncommitted files.

```powershell
$releaseRoot = 'H:\hypura-release-artifacts\v1.0.0'
python .\scripts\package_fullsource.py `
  --version 1.0.0 `
  --commit HEAD `
  --output-dir $releaseRoot
if ($LASTEXITCODE -ne 0) { throw 'first fullsource build failed' }

$fullsource = "$releaseRoot\hypura-v1.0.0-fullsource.tar.gz"
$firstHash = (Get-FileHash -LiteralPath $fullsource -Algorithm SHA256).Hash
python .\scripts\package_fullsource.py `
  --version 1.0.0 `
  --commit HEAD `
  --output-dir $releaseRoot
if ($LASTEXITCODE -ne 0) { throw 'second fullsource build failed' }
$secondHash = (Get-FileHash -LiteralPath $fullsource -Algorithm SHA256).Hash
if ($firstHash -ne $secondHash) { throw 'fullsource archive is not deterministic' }
```

Extract the archive to a fresh directory, inspect `SOURCE-MANIFEST.json`, and run at least `cargo metadata --no-deps` from the extracted tree. A stable release should also run the appropriate build smoke test from that extracted source.

## 7. Stage both asset sets

```powershell
$cli = 'H:\hypura-cargo-target-v1.0.0\release\Hypura.exe'
$fullsource = "$releaseRoot\hypura-v1.0.0-fullsource.tar.gz"
$msi = 'H:\hypura-desktop-release-target-v1.0.0\release\bundle\msi\Hypura Desktop_1.0.0_x64_en-US.msi'
$nsis = 'H:\hypura-desktop-release-target-v1.0.0\release\bundle\nsis\Hypura Desktop_1.0.0_x64-setup.exe'

python .\scripts\stage_release_assets.py `
  --version 1.0.0 --channel stable `
  --cli $cli --fullsource $fullsource --msi $msi --nsis $nsis `
  --output-dir "$releaseRoot\stable"
if ($LASTEXITCODE -ne 0) { throw 'stable asset staging failed' }

python .\scripts\stage_release_assets.py `
  --version 1.0.0 --channel main `
  --cli $cli --fullsource $fullsource --msi $msi --nsis $nsis `
  --output-dir "$releaseRoot\main"
if ($LASTEXITCODE -ne 0) { throw 'main asset staging failed' }
```

Verify every line of each `SHA256SUMS.txt` before publication.

## 8. Overwrite installation

Only the verified release executable may overwrite existing installations.

```powershell
$repoInstall = Join-Path $env:USERPROFILE 'Desktop\hypura-main\hypura-main\target\release\Hypura.exe'
$cargoInstall = Join-Path $env:USERPROFILE '.cargo\bin\Hypura.exe'
Copy-Item -LiteralPath $cli -Destination $repoInstall -Force
Copy-Item -LiteralPath $cli -Destination $cargoInstall -Force

$expectedHash = (Get-FileHash -LiteralPath $cli -Algorithm SHA256).Hash
foreach ($installedCli in @($repoInstall, $cargoInstall)) {
  $actualHash = (Get-FileHash -LiteralPath $installedCli -Algorithm SHA256).Hash
  if ($actualHash -ne $expectedHash) {
    throw "Installed CLI hash mismatch: $installedCli"
  }
  $installedVersion = (& $installedCli --version | Out-String)
  if ($LASTEXITCODE -ne 0 -or $installedVersion -notmatch '\b1\.0\.0\b') {
    throw "Installed CLI version mismatch: $installedCli"
  }
}
```

Install and verify both Desktop package formats. Record whether the first package upgraded an existing installation or performed a first installation, then confirm that the second package preserves version 1.0.0.

```powershell
function Get-HypuraDesktopInstall {
  $roots = @(
    'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall',
    'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall',
    'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall'
  )
  foreach ($root in $roots) {
    if (Test-Path -LiteralPath $root) {
      Get-ChildItem -LiteralPath $root | ForEach-Object {
        Get-ItemProperty -LiteralPath $_.PSPath
      } | Where-Object { $_.DisplayName -eq 'Hypura Desktop' }
    }
  }
}

$beforeDesktop = @(Get-HypuraDesktopInstall | Select-Object -First 1)
& msiexec.exe /i $msi /qn /norestart
$msiExitCode = $LASTEXITCODE
if ($msiExitCode -notin @(0, 3010)) { throw "MSI install failed: $msiExitCode" }
$afterMsi = @(Get-HypuraDesktopInstall | Where-Object {
  $_.DisplayVersion -eq '1.0.0' -and (
    $_.WindowsInstaller -eq 1 -or $_.UninstallString -match '(?i)msiexec'
  )
})
if ($afterMsi.Count -ne 1) {
  throw 'MSI product version 1.0.0 was not registered'
}

$nsisInstall = Start-Process -FilePath $nsis -ArgumentList '/S' `
  -WindowStyle Hidden -Wait -PassThru
if ($nsisInstall.ExitCode -ne 0) { throw "NSIS install failed: $($nsisInstall.ExitCode)" }
$afterNsis = @(Get-HypuraDesktopInstall | Where-Object {
  $_.DisplayVersion -eq '1.0.0' -and
  $_.WindowsInstaller -ne 1 -and
  $_.UninstallString -match '(?i)\.exe'
})
if ($afterNsis.Count -ne 1) {
  throw 'NSIS product version 1.0.0 was not registered'
}

$upgradeState = if ($beforeDesktop.Count -eq 1) { 'upgrade' } else { 'first_install' }
[pscustomobject]@{
  state = $upgradeState
  previous_version = if ($beforeDesktop.Count) { $beforeDesktop[0].DisplayVersion } else { $null }
  installed_version = $afterNsis[0].DisplayVersion
  reboot_required = ($msiExitCode -eq 3010)
  msi_exit_code = $msiExitCode
  msi_uninstall = $afterMsi[0].UninstallString
  nsis_uninstall = $afterNsis[0].UninstallString
} | ConvertTo-Json | Set-Content -LiteralPath "$releaseRoot\desktop-install-proof.json" -Encoding UTF8
```

## 9. Publish stable and main

Push the tested release commit to `main`, then create the stable branch and both annotated tags at that exact commit.

```powershell
$releaseSha = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'release SHA resolution failed' }

git push origin "${releaseSha}:refs/heads/main"
if ($LASTEXITCODE -ne 0) { throw 'main push failed; publication stopped' }
git push origin "${releaseSha}:refs/heads/stable/v1.0.0"
if ($LASTEXITCODE -ne 0) { throw 'stable branch push failed; publication stopped' }
git tag -a v1.0.0 $releaseSha -m 'release v1.0.0'
if ($LASTEXITCODE -ne 0) { throw 'stable tag creation failed; publication stopped' }
git tag -a main-v1.0.0 $releaseSha -m 'main snapshot v1.0.0'
if ($LASTEXITCODE -ne 0) { throw 'main tag creation failed; publication stopped' }
git push origin v1.0.0 main-v1.0.0
if ($LASTEXITCODE -ne 0) { throw 'tag push failed; publication stopped' }
```

```powershell
$stableAssets = @(
  "$releaseRoot\stable\hypura-v1.0.0-windows-x86_64-sm120.exe"
  "$releaseRoot\stable\hypura-v1.0.0-fullsource.tar.gz"
  "$releaseRoot\stable\hypura-v1.0.0-fullsource.tar.gz.sha256"
  "$releaseRoot\stable\hypura-desktop-v1.0.0-windows-x64.msi"
  "$releaseRoot\stable\hypura-desktop-v1.0.0-windows-x64-setup.exe"
  "$releaseRoot\stable\SHA256SUMS.txt"
)

$mainAssets = @(
  "$releaseRoot\main\hypura-main-v1.0.0-windows-x86_64-sm120.exe"
  "$releaseRoot\main\hypura-main-v1.0.0-fullsource.tar.gz"
  "$releaseRoot\main\hypura-main-v1.0.0-fullsource.tar.gz.sha256"
  "$releaseRoot\main\hypura-desktop-main-v1.0.0-windows-x64.msi"
  "$releaseRoot\main\hypura-desktop-main-v1.0.0-windows-x64-setup.exe"
  "$releaseRoot\main\SHA256SUMS.txt"
)

$installerSignatures = Get-AuthenticodeSignature -LiteralPath $msi,$nsis
$installerStatus = ($installerSignatures | ForEach-Object {
  "$(Split-Path -Leaf $_.Path): $($_.Status)"
}) -join '; '

function Get-GitlinkSha([string]$Repo, [string]$Commit, [string]$Path) {
  $entry = (git -C $Repo ls-tree $Commit -- $Path | Out-String).Trim()
  if ($LASTEXITCODE -ne 0 -or -not $entry) {
    throw "gitlink resolution failed: $Repo $Commit $Path"
  }
  $parts = $entry -split '\s+', 4
  if ($parts.Count -lt 3 -or $parts[0] -ne '160000') {
    throw "expected gitlink: $Repo $Commit $Path"
  }
  $parts[2]
}

$llamaSha = Get-GitlinkSha . $releaseSha 'vendor/llama.cpp'
$turboquantSha = Get-GitlinkSha . $releaseSha 'vendor/turboquant-cuda'
$nestedLlamaSha = Get-GitlinkSha vendor/turboquant-cuda $turboquantSha 'zapabob/llama.cpp'

$stableNotes = @"
Hypura v1.0.0 is the stable Windows CUDA 12.8 release for RTX 50-series sm_120 GPUs.

This release adds the dedicated three-candidate Answer Council, strict schema-v2 Triality contracts, finite-moment NC-KA evaluation, URT consistency records, and opt-in Aha event reporting. Attention-logit consensus remains a separate low-level mode. The identity-view QA fixture is wiring evidence only and is not a model-quality claim.

Source pins: Hypura $releaseSha; llama.cpp $llamaSha; Turboquant-CUDA $turboquantSha; nested llama.cpp $nestedLlamaSha.

$installerStatus Verify every download with SHA256SUMS.txt before installation.
"@
$mainNotes = @"
Hypura main snapshot v1.0.0 contains the same tested source commit and binaries as the stable v1.0.0 release, under main-channel asset names.

Source pins: Hypura $releaseSha; llama.cpp $llamaSha; Turboquant-CUDA $turboquantSha; nested llama.cpp $nestedLlamaSha.

$installerStatus Verify every download with SHA256SUMS.txt before installation.
"@
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText("$releaseRoot\release-notes-stable.md", $stableNotes, $utf8NoBom)
[System.IO.File]::WriteAllText("$releaseRoot\release-notes-main.md", $mainNotes, $utf8NoBom)

gh release create v1.0.0 `
  -R zapabob/hypura `
  --target stable/v1.0.0 `
  --verify-tag `
  --latest `
  --title 'Hypura v1.0.0 Stable' `
  --notes-file "$releaseRoot\release-notes-stable.md" `
  @stableAssets
if ($LASTEXITCODE -ne 0) { throw 'stable GitHub release creation failed' }

gh release create main-v1.0.0 `
  -R zapabob/hypura `
  --target main `
  --verify-tag `
  --latest=false `
  --title 'Hypura main snapshot v1.0.0' `
  --notes-file "$releaseRoot\release-notes-main.md" `
  @mainAssets
if ($LASTEXITCODE -ne 0) { throw 'main GitHub release creation failed' }
```

## 10. Post-publication proof

```powershell
$remoteLines = @(git ls-remote --heads --tags origin `
  refs/heads/main `
  refs/heads/stable/v1.0.0 `
  refs/tags/v1.0.0 `
  'refs/tags/v1.0.0^{}' `
  refs/tags/main-v1.0.0 `
  'refs/tags/main-v1.0.0^{}')
if ($LASTEXITCODE -ne 0) { throw 'remote reference verification failed' }
$remoteRefs = @{}
foreach ($line in $remoteLines) {
  $sha, $ref = $line -split "`t", 2
  if ($sha -and $ref) { $remoteRefs[$ref] = $sha.ToLowerInvariant() }
}
$releaseSha = $releaseSha.ToLowerInvariant()
foreach ($ref in @(
  'refs/heads/main',
  'refs/heads/stable/v1.0.0',
  'refs/tags/v1.0.0^{}',
  'refs/tags/main-v1.0.0^{}'
)) {
  if ($remoteRefs[$ref] -ne $releaseSha) {
    throw "Remote ref does not resolve to the release commit: $ref"
  }
}
foreach ($tag in @('refs/tags/v1.0.0', 'refs/tags/main-v1.0.0')) {
  if (-not $remoteRefs.ContainsKey($tag) -or $remoteRefs[$tag] -eq $releaseSha) {
    throw "Annotated tag object is missing: $tag"
  }
}

$stableRelease = gh release view v1.0.0 -R zapabob/hypura `
  --json tagName,targetCommitish,isDraft,isPrerelease,publishedAt,url,assets |
  ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'stable release verification failed' }

$mainRelease = gh release view main-v1.0.0 -R zapabob/hypura `
  --json tagName,targetCommitish,isDraft,isPrerelease,publishedAt,url,assets |
  ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'main release verification failed' }

function Assert-ReleaseMetadata(
  $Release,
  [string]$Tag,
  [string]$Target,
  [string[]]$ExpectedAssets
) {
  if (
    $Release.tagName -ne $Tag -or
    $Release.targetCommitish -ne $Target -or
    $Release.isDraft -or
    $Release.isPrerelease -or
    -not $Release.publishedAt
  ) {
    throw "Release metadata mismatch: $Tag"
  }
  $actualAssets = @($Release.assets.name | Sort-Object)
  $expected = @($ExpectedAssets | Sort-Object)
  if (Compare-Object $expected $actualAssets) {
    throw "Release asset set mismatch: $Tag"
  }
}

$stableNames = @(
  'hypura-v1.0.0-windows-x86_64-sm120.exe',
  'hypura-v1.0.0-fullsource.tar.gz',
  'hypura-v1.0.0-fullsource.tar.gz.sha256',
  'hypura-desktop-v1.0.0-windows-x64.msi',
  'hypura-desktop-v1.0.0-windows-x64-setup.exe',
  'SHA256SUMS.txt'
)
$mainNames = @(
  'hypura-main-v1.0.0-windows-x86_64-sm120.exe',
  'hypura-main-v1.0.0-fullsource.tar.gz',
  'hypura-main-v1.0.0-fullsource.tar.gz.sha256',
  'hypura-desktop-main-v1.0.0-windows-x64.msi',
  'hypura-desktop-main-v1.0.0-windows-x64-setup.exe',
  'SHA256SUMS.txt'
)
Assert-ReleaseMetadata $stableRelease 'v1.0.0' 'stable/v1.0.0' $stableNames
Assert-ReleaseMetadata $mainRelease 'main-v1.0.0' 'main' $mainNames

$downloadProof = Join-Path $releaseRoot "download-proof-$releaseSha"
if (Test-Path -LiteralPath $downloadProof) {
  throw "Download proof path already exists: $downloadProof"
}
$null = New-Item -ItemType Directory -Path $downloadProof
$stableDownload = New-Item -ItemType Directory -Path "$downloadProof\stable"
$mainDownload = New-Item -ItemType Directory -Path "$downloadProof\main"
gh release download v1.0.0 -R zapabob/hypura --dir $stableDownload.FullName
if ($LASTEXITCODE -ne 0) { throw 'stable release download failed' }
gh release download main-v1.0.0 -R zapabob/hypura --dir $mainDownload.FullName
if ($LASTEXITCODE -ne 0) { throw 'main release download failed' }

function Assert-Sha256Manifest([string]$Directory, $Release) {
  $manifest = Join-Path $Directory 'SHA256SUMS.txt'
  $downloadedNames = @(
    Get-ChildItem -LiteralPath $Directory -File | ForEach-Object Name | Sort-Object
  )
  $releaseNames = @($Release.assets.name | Sort-Object)
  if (Compare-Object $releaseNames $downloadedNames) {
    throw "Downloaded asset set mismatch: $Directory"
  }

  $entries = @()
  foreach ($line in Get-Content -LiteralPath $manifest) {
    $parts = @($line.Trim() -split '\s+', 2)
    if ($parts.Count -ne 2 -or $parts[0] -notmatch '^[0-9a-fA-F]{64}$') {
      throw "Malformed checksum entry: $line"
    }
    $expectedHash, $name = $parts
    $name = $name.Trim()
    if ([System.IO.Path]::GetFileName($name) -ne $name) {
      throw "Unsafe checksum entry: $name"
    }
    $entries += [pscustomobject]@{ hash = $expectedHash; name = $name }
  }

  $manifestNames = @($entries.name | Sort-Object)
  if (@($manifestNames | Select-Object -Unique).Count -ne $manifestNames.Count) {
    throw "Duplicate checksum entry: $manifest"
  }
  $expectedManifestNames = @($releaseNames | Where-Object { $_ -ne 'SHA256SUMS.txt' })
  if (Compare-Object $expectedManifestNames $manifestNames) {
    throw "Checksum manifest asset coverage mismatch: $manifest"
  }

  foreach ($entry in $entries) {
    $name = $entry.name
    $assetPath = Join-Path $Directory $name
    $actualHash = (Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $entry.hash.ToLowerInvariant()) {
      throw "Downloaded release hash mismatch: $name"
    }
  }

  foreach ($asset in $Release.assets) {
    if ([string]::IsNullOrWhiteSpace($asset.digest)) {
      throw "GitHub asset digest is missing: $($asset.name)"
    }
    $assetPath = Join-Path $Directory $asset.name
    $actualHash = (Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($asset.digest.ToLowerInvariant() -ne "sha256:$actualHash") {
      throw "GitHub asset digest mismatch: $($asset.name)"
    }
  }
}
Assert-Sha256Manifest $stableDownload.FullName $stableRelease
Assert-Sha256Manifest $mainDownload.FullName $mainRelease

$downloadedCli = Join-Path $stableDownload.FullName 'hypura-v1.0.0-windows-x86_64-sm120.exe'
$downloadedVersion = (& $downloadedCli --version | Out-String)
if ($LASTEXITCODE -ne 0 -or $downloadedVersion -notmatch '\b1\.0\.0\b') {
  throw 'downloaded stable CLI version verification failed'
}
& $downloadedCli --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'downloaded stable CLI help verification failed' }
```
