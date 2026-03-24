#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2048 で Hypura を一時起動し、ヘルス + 短い /api/generate が通ったら停止して
central-state.json を 8192 に更新する（tqdm で待機を可視化）。

Caption: Smoke test and promote context to 8192 with progress bar.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def tqdm_range(n: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(range(n), desc=desc, unit="try", ascii=True)
    except Exception:
        return range(n)


def wait_http_ok(base: str, max_wait: int, delay_sec: float) -> None:
    health = base.rstrip("/") + "/"
    for _ in tqdm_range(max_wait, "wait hypura /"):
        try:
            req = urllib.request.Request(health, method="GET")
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(delay_sec)
    raise SystemExit("timeout: Hypura did not respond on GET /")


def get_model_name(base: str) -> str:
    url = base.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    models = data.get("models") or []
    if not models:
        raise SystemExit("no models in /api/tags")
    return str(models[0].get("name") or models[0].get("model"))


def post_generate(base: str, model: str) -> None:
    url = base.rstrip("/") + "/api/generate"
    body = json.dumps(
        {
            "model": model,
            "prompt": "Say OK.",
            "stream": False,
            "options": {"num_predict": 8},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 200:
            raise SystemExit(f"generate status {r.status}")


def write_state(path: Path, context: int, tier: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "context": context,
        "tier": tier,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kv_compact_note": (
            "8192: placement splits hot/warm KV; KvCacheManager compacts past hot window"
        ),
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Hypura smoke then promote state to 8192")
    p.add_argument("--exe", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--state-path", type=Path, required=True)
    p.add_argument("--smoke-context", type=int, default=2048)
    p.add_argument("--promote-context", type=int, default=8192)
    p.add_argument("--max-wait", type=int, default=180, help="attempts for health check")
    p.add_argument("--delay", type=float, default=1.0)
    args = p.parse_args()

    base = f"http://{args.host}:{args.port}"

    # quick port check: if something already listens, bail
    try:
        urllib.request.urlopen(base + "/", timeout=2)
        raise SystemExit(
            f"refusing: something already responds on {base} (stop it or change port)"
        )
    except Exception:
        pass  # connection refused / timeout → OK to start our process

    cmd = [
        args.exe,
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--context",
        str(args.smoke_context),
    ]
    print("[promote_smoke] starting:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)  # noqa: SIM115

    try:
        wait_http_ok(base, args.max_wait, args.delay)
        name = get_model_name(base)
        print("[promote_smoke] model:", name, flush=True)
        post_generate(base, name)
        print("[promote_smoke] generate OK", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    write_state(args.state_path, args.promote_context, "full")
    print("[promote_smoke] wrote", args.state_path, "context=", args.promote_context, flush=True)


if __name__ == "__main__":
    main()
