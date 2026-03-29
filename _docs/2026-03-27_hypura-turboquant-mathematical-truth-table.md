# Hypura TurboQuant 数理的真理値表

- 日付: 2026-03-27
- 対象リポジトリ: `hypura-main`
- 対象モデル: `Qwen3.5-9B`
- 想定実装系: Hugging Face Transformers
- 実行環境: Windows, RTX 3060
- 前提 runtime: offline-first, online generation は補助評価, production default は key-only

TurboQuant を `Hypura` に拡張するうえで重要なのは、論文の主張を要約することではなく、「何が本当に成り立ち、何が成り立たず、何が条件付きか」を切り分けることです。以下は、TurboQuant 論文、`Turboquant-CUDA` README、そして Qwen3.5-9B に対する既存観測を前提に、Hypura の default / experimental / reject を決めるための判定表です。

## 判定規則

- `真`: 一次資料と観測が概ね一致し、強い反例がない。
- `偽`: 観測または数理上の反例が強く、Hypura の default 前提にしてはいけない。
- `条件付き`: 限定 regime では成立するが、`Qwen3.5-9B + HF + Windows + RTX3060 + offline-first` へは一般化できない。
- `未確定`: 重要だが、現状の観測だけでは切れない。
- `推論`: 論文や README に直接は書かれていないが、合理的に導いた内容。

補足:

- `推論` のみで支える命題は `真` にしない。最大でも `条件付き`。
- 根拠ラベルは `論文`, `Turboquant-CUDA`, `観測`, `推論` に限定する。

## 1. 数学的命題

| 命題 | 判定 | 根拠 | 数理的理由 | Hypura 設計帰結 | 次に潰すべき実験 |
|---|---|---|---|---|---|
| 固定ランダム回転で実 KV 分布は十分に等方化される | 偽 | 論文, 推論, 観測 | 固定 `R` では `\Sigma_z = R \Sigma_x R^\top` で固有値は不変。向きは変わるが anisotropy 自体は消えない。論文の Beta 的座標は `x` 固定・`R` ランダムの議論で、実装の `R` 固定・`x ~ P_{KV}` とは別物。 | random rotation を普遍前提にしない。V では blockwise/no-rotation を候補化する。 | `no-rotation / random-rotation / block-rotation` の V アブレーション |
| unit-sphere 上の near-optimality は norm-separated 実 KV にそのまま移る | 条件付き | 論文, 推論 | `x = ||x||u` にすると `\delta x = (||x||-\hat{||x||})u + ||x||(u-\hat{u})`。direction-only の最適性は radial error を吸収しない。heavy norm token では絶対誤差が増幅される。 | norm は cheap side-channel として別管理する設計を前提化する。 | norm exact / low-bit / shared-scale 比較 |
| inner-product unbiasedness は softmax ranking preservation に十分である | 偽 | 論文, 推論, 観測 | `E[\hat{\langle q,k\rangle}] = \langle q,k\rangle` でも top-rank 保全には `||\varepsilon||_\infty < (\ell_{(1)}-\ell_{(2)})/2` が要る。unbiased は平均でしかなく、margin が小さい token では順位反転する。 | K の評価は平均誤差だけでなく margin flip rate を入れる。 | top-1 / top-k margin flip 率の seed sweep |
| 小さい MSE は decoding stability に十分である | 偽 | 論文, 観測, 推論 | MSE は `\min E||x-\hat{x}||^2` だが、欲しいのは downstream-sensitive 誤差。観測では logit cosine が近くても hidden-state cosine が full-KV で崩れる。目的関数が違う。 | V codec を MSE 最適化の延長で作らない。 | hidden-state sensitivity 損失での V codec 比較 |
| `\Delta O = A(\hat{V}-V)` は attention averaging により十分に抑えられる | 偽 | 観測, 推論 | 1 step では `||\Delta O|| \le \max_j ||e_j||` だが、危険なのは bias/covariance と autoregressive recurrence。`\Delta O` が residual stream に直接注入され、層と時間で累積する。 | V は transport codec として別設計。full-KV を default にしない。 | `A(\hat{V}-V)` ノルムと layerwise hidden divergence の相関計測 |
| heavy-tail / anisotropy / outlier regime でも random rotation は無害である | 偽 | 推論, 観測 | outlier / sensitive channel が元 basis に sparse にあるなら、random rotation はそれを全座標へ拡散し、cheap な保護をむしろ難しくする。 | protected-V をやるなら protected subspace と rest を混ぜない。 | protected-channel + full rotation vs block rotation |
| fixed seed / fixed sketch の pathwise failure は無視できる | 偽 | 推論 | 実運用は `R, S` を 1 回固定して使う。必要なのは expectation ではなく pathwise / worst-seed 安定性。 | seed sweep を評価 protocol に組み込む。 | `R, S` 3-10 seed で worst-case hidden / logit 監査 |

## 2. アルゴリズム命題

| 命題 | 判定 | 根拠 | 数理的理由 | Hypura 設計帰結 | 次に潰すべき実験 |
|---|---|---|---|---|---|
| K と V に同型 codec を使ってよい | 偽 | Turboquant-CUDA, 観測, 推論 | K は `q^\top k` の preservation、V は `O = AV` の transport preservation。objective が違う。`Turboquant-CUDA` README も full-KV は value transport を先に壊すと述べる。 | K/V 分離を設計原則にする。 | K-only / V-only / full-KV 分解アブレーション |
| TurboQuantProd の residual QJL は V にも自然である | 偽 | 論文, 観測, 推論 | QJL residual は scalar inner-product 補正には自然だが、V は vector transport 問題。1-bit residual sketch は hidden geometry を戻しにくい。 | V には Prod を既定採用しない。 | V: Prod vs MSE-only vs exact |
| random rotation は V 保護設計と両立する | 条件付き | 推論 | full dense rotation のまま protected channel を選ぶと、保護の意味が消える。blockwise / masked rotation なら両立余地がある。 | protected-V をやるなら `I \oplus R_{rest}` 型に限定。 | protected-V with block rotation |
| mixed-bit outlier 保護は calibration-free でも十分である | 偽 | 論文, Turboquant-CUDA, 推論 | magnitude-based outlier 保護は downstream sensitivity を見ない。大きいが鈍感な channel を守り、本当に危険な channel を落としうる。 | V 保護は calibration-driven に寄せる。 | magnitude top-k vs sensitivity top-k 比較 |
| global bit policy で layer/head heterogeneity は無視できる | 偽 | 観測, 推論 | 感度は layer/head でかなり不均一なはず。global 2.5 / 3.5 / 4.0 は鈍感層に無駄打ちし、敏感層を starving する。 | adaptive bit allocation を experimental 候補に置く。 | layer/head 感度 map |
| full-KV を baseline path にしてよい | 偽 | Turboquant-CUDA, 観測 | README も Qwen replay で full-KV が hidden transport を弱く保つと述べる。runtime 観測でも key-only が安定。 | `paper-full-kv` は validation only。 | full-KV を baseline 扱いした時の online prefix 崩壊監査 |
| key-only は妥協案にすぎず、最終形ではない | 条件付き | 観測, 推論 | 現時点では実務最適に見えるが、研究的には V の別 codec で Pareto 改善余地がある。だから「最終形ではない」は弱く真だが、近未来 default としては最良。 | default は key-only、研究は V 別 codec。 | protected-V / low-rank V residual |

## 3. 実装命題

| 命題 | 判定 | 根拠 | 数理的理由 | Hypura 設計帰結 | 次に潰すべき実験 |
|---|---|---|---|---|---|
| offline logit 指標で generation 品質を十分に代理できる | 偽 | Turboquant-CUDA, 観測 | README でも logit-like 指標と hidden transport が分離。観測でも logit cosine は近いのに full-KV の hidden と generation が崩れる。 | logit だけで合格にしない。 | next-logit KL + hidden cosine + prefix match の同時計測 |
| hidden-state cosine が高ければ online generation も安定する | 条件付き | 観測, 推論 | 必要条件寄りだが十分条件ではない。small hidden drift でも early token mismatch が recurrence で増幅される。 | online 補助評価を残す。 | teacher-forced hidden と free-running divergence の比較 |
| V 誤差は局所誤差であり、層を跨いだ累積は小さい | 偽 | 観測, 推論 | `||\delta h_{\ell+1}|| \le L_\ell ||\delta h_\ell|| + ||B_{\ell,V}\delta V_\ell|| + ||B_{\ell,K}\delta K_\ell||`。観測上、主因は `\delta V_\ell` 側。これが residual で積み上がる。 | layerwise fail-fast 監視を入れる。 | layerwise hidden divergence heatmap |
| metadata / protection mask の byte overhead は主問題である | 偽 | 推論 | static mask 自体は `O(LHd)` bits で小さい。主問題は gather/scatter と kernel overhead、exact subset buffer。 | metadata ではなく runtime kernel 設計を気にする。 | mask bytes と kernel time の分離 profile |
| runtime latency の主因は codec の算術量である | 条件付き | 推論 | Windows + HF では small kernel / host sync / memory traffic も大きい。V は dequant/reconstruct の形次第で launch overhead が支配しうる。 | V codec は FLOPs より kernel shape を意識。 | torch.profiler で launch / memcpy / matmul 分解 |
| HF + RTX3060 + Windows でも paper-faithful full-KV は実務的である | 偽 | 観測, 推論 | 品質面でも runtime 面でもリスクが高い。少なくとも現観測では production default にする材料がない。 | Hypura でも full-KV は opt-in experimental。 | long prompt / online chat の failure rate 監査 |

## 4. Hypura 設計命題

| 命題 | 判定 | 根拠 | 数理的理由 | Hypura 設計帰結 | 次に潰すべき実験 |
|---|---|---|---|---|---|
| Hypura の production default は key-only でよい | 真 | 観測, Turboquant-CUDA, 推論 | 現時点で最も強い実証がある。K 側は目的と理論がまだ近く、V 側が主破綻源。 | default = `paper-key-only`。 | key-only の seed / context / prompt family 監査 |
| `paper-full-kv` は validation path に留めるべきである | 真 | 観測, Turboquant-CUDA | faithful baseline としての価値はあるが、runtime default に置く根拠がない。 | `paper-full-kv` は benchmark / ablation only。 | full-KV regression suite |
| V は別 codec に分離すべきである | 真 | 観測, Turboquant-CUDA, 推論 | K/V objective mismatch が本質。README も K/V-separated research extension を明示。 | `K = Prod`, `V = separate codec` を roadmap 主軸にする。 | V codec family ablation |
| imatrix 的 V protection は条件付きで採用価値がある | 条件付き | 推論, Turboquant-CUDA | sensitivity-based なら有望。magnitude-based なら筋が悪い。protected 部分と full rotation の併用も危険。 | experimental に置く。 | sensitivity top-k `\alpha` sweep |
| TurboQuant の faithful reproduction より K/V 分離設計のほうが価値が高い | 真 | 観測, Turboquant-CUDA, 推論 | faithful full-KV を詰めても、壊れている仮定自体は直らない。 | Hypura では paper baseline を通過点扱いにする。 | K/V 分離 hybrid の Pareto 比較 |
| Hypura に必要なのは GGUF 内埋め込みではなく sidecar artifact である | 真 | repo 現実装, 推論 | K/V 分離・protected mask・adaptive policy は sidecar のほうが更新容易。GGUF 内埋め込みは仕様硬化が早すぎる。 | config/artifact は sidecar 継続。 | sidecar schema で V protection artifact を持つ実験 |

## Hypura で default にしてよいもの

- `K = TurboQuantProd mixed-bit`, `V = exact`
- sidecar artifact 運用
- `paper-full-kv` 非default
- logit 指標に hidden / transport / prefix 指標を併記する評価

## Hypura で experimental に留めるもの

- `paper-full-kv`
- static sensitivity-weighted V protection
- per-layer / per-head adaptive bit allocation
- low-rank residual correction for V
- learned / blockwise rotation

## Hypura で捨てるべき前提

- `K/V は同型 codec でよい`
- `MSE が小さければ generation も安定する`
- `unbiased inner product なら softmax も safe`
- `full-KV faithful reproduction が最優先`

## 最重要 5 実験

### 1. `V codec family` 分解アブレーション

- 仮説: 壊れているのは V 量子化一般ではなく、`Prod + random rotation` の組み合わせ。
- 実装難易度: 中
- メモリ影響: 比較的固定
- runtime 影響: 中
- falsification 条件: V のどの codec でも同程度に崩れるならこの仮説は棄却
- priority: 最高

### 2. layer/head V sensitivity map

- 仮説: V 感度は層・頭に集中している。
- 実装難易度: 中
- メモリ影響: 低
- runtime 影響: 中
- falsification 条件: 感度が一様なら selective protection は効きにくい
- priority: 最高

### 3. static sensitivity-weighted V protection `\alpha`-sweep

- 仮説: `\alpha = 0.1–0.3` の exact/high-bit 保護で大きな Pareto 改善がある。
- 実装難易度: 中
- メモリ影響: `R(\alpha) \approx R_{KV} + \alpha(R_K - R_{KV})`
- runtime 影響: 低〜中
- falsification 条件: 少量保護で hidden / generation が戻らない
- priority: 高

### 4. per-layer / per-head adaptive bit allocation

- 仮説: global 3.5bit より同メモリで明確に良くなる。
- 実装難易度: 中
- メモリ影響: 予算固定
- runtime 影響: 低〜中
- falsification 条件: adaptive にしても global と差がほぼない
- priority: 高

### 5. low-rank residual correction for V

- 仮説: `v_j \approx \hat{v}_j + U c_j` の rank 4–8 で hidden transport が大きく戻る。
- 実装難易度: 中〜高
- メモリ影響: 小
- runtime 影響: 低〜中
- falsification 条件: residual spectrum が flat で low-rank が効かない
- priority: 中〜高

## 最終判断

- 最も本質的な conceptual flaw は、`K` と `V` を同じ幾何・同じ objective で圧縮してよい、という前提です。
- 近い将来の実務上の最善策は、`K = TurboQuantProd mixed-bit`, `V = exact` を標準にし、V は selective protection か別 codec で段階的に攻めることです。
- ハイリスクだが研究価値が高い方向は、`K = TurboQuantProd`, `V = transport-aware codec` の分離設計、特に sensitivity-weighted protection と low-rank residual correction です。

## Sources

- TurboQuant 論文: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)
- `Turboquant-CUDA` README: [GitHub raw README](https://raw.githubusercontent.com/zapabob/Turboquant-CUDA/main/README.md)
