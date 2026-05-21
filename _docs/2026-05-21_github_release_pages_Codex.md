# 2026-05-21 GitHub release and Pages publication

## Scope

Publish Hypura v0.15.0 to GitHub with both stable and main release surfaces, and
publish a GitHub Pages product site based on a generated hero image.

## Branches

- `main`: `7b4a3ba45751d0e6e5a0168248d3406292770294`
- `stable/v0.15.0`: `7b4a3ba45751d0e6e5a0168248d3406292770294`
- `gh-pages`: `d22bd41c49b6adabf5589a0da2f38d076fe0563f`

## Releases

- Stable release: https://github.com/zapabob/hypura/releases/tag/v0.15.0
  - Target: `stable/v0.15.0`
  - Asset: `hypura-v0.15.0-stable-fullsource.tar.gz`
  - SHA256: `42BD9A6C53CEF995F37A29BC8DA61A4AFAAD5BC440B3C0472BC7BC7B3A5E6F5B`
- Main snapshot release: https://github.com/zapabob/hypura/releases/tag/main-v0.15.0
  - Target: `main`
  - Asset: `hypura-main-7b4a3ba-fullsource.tar.gz`
  - SHA256: `42BD9A6C53CEF995F37A29BC8DA61A4AFAAD5BC440B3C0472BC7BC7B3A5E6F5B`

Both tarballs are full-source snapshots with vendored submodule contents. Local
copies were staged under `H:\hypura-release-artifacts\v0.15.0`.

## Pages

- Published URL: https://zapabob.github.io/hypura/
- Source branch: `gh-pages`
- Build status: `built`
- Hero asset: `assets/hypura-hero.png`
- Hero image SHA256:
  `BB73EAA108F47DA0D55A4997D535B281E2ECD6458B0767F12716AEFBE089FC19`

The generated image was produced with the built-in image generation path and
copied into the Pages branch as a project-local asset.

## Verification

```powershell
git submodule status --recursive
cargo metadata --no-deps --format-version 1
gh release list -R zapabob/hypura --limit 5
gh api repos/zapabob/hypura/pages/builds/latest
Invoke-WebRequest -Uri https://zapabob.github.io/hypura/ -UseBasicParsing -TimeoutSec 20
Invoke-WebRequest -Uri https://zapabob.github.io/hypura/assets/hypura-hero.png -UseBasicParsing -TimeoutSec 20
```

Observed:

- `cargo metadata` reported `hypura=0.15.0` and `hypura-sys=0.15.0`.
- `v0.15.0` was marked Latest in GitHub Releases.
- Pages returned HTTP 200 for both `index.html` and `assets/hypura-hero.png`.
- Desktop and mobile screenshots were rendered locally with headless Chrome
  before publication.
