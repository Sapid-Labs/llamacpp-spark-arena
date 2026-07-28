#!/usr/bin/env python3
"""
llama.cpp Spark arena harness.

Paired-ratio measurement with a token-identity correctness gate, for
kernel/graph optimization work on DGX Spark (GB10, sm_121).

The shape of a run:

    baseline arm -> candidate arm -> baseline arm -> candidate arm ...

alternating, back to back, on one node, at batch 1, with the first request
after every server start discarded. The ratio -- not the absolute tok/s --
is the measurement, because absolute numbers on this hardware do not survive
a thermal or host change (a published Laguna baseline failed to reproduce a
day later: 24.05 -> 20.09 tok/s).

Correctness is a token diff, not an eval. At batch 1, warm, greedy, this
engine is deterministic (24/24 identical, measured 2026-07-28), so
"did this change alter the output?" is answerable for free. Any batching
breaks that, which is why the ranked track is serial.

Stdlib only. Runs on the node, over SSH or locally.

    python3 harness/arena.py baseline --target <t>          # record goldens + baseline
    python3 harness/arena.py bench    --target <t>          # paired candidate vs baseline
    python3 harness/arena.py gate     --target <t>          # editable-surface check only
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = json.loads((ROOT / "benchmark.json").read_text())

BASE_TREE = ROOT / "vendor" / "llama.cpp-base"      # pristine, at the pin
CAND_TREE = ROOT / "vendor" / "llama.cpp"           # yours, editable


# --------------------------------------------------------------------- output

class Out:
    BOLD, DIM, RED, GRN, YEL, OFF = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"

    @staticmethod
    def step(msg):
        print(f"{Out.BOLD}==>{Out.OFF} {msg}", flush=True)

    @staticmethod
    def info(msg):
        print(f"    {Out.DIM}{msg}{Out.OFF}", flush=True)

    @staticmethod
    def ok(msg):
        print(f"    {Out.GRN}PASS{Out.OFF} {msg}", flush=True)

    @staticmethod
    def fail(msg):
        print(f"    {Out.RED}FAIL{Out.OFF} {msg}", flush=True)

    @staticmethod
    def warn(msg):
        print(f"    {Out.YEL}warn{Out.OFF} {msg}", flush=True)


def die(msg, code=1):
    print(f"\n{Out.RED}{Out.BOLD}arena: {msg}{Out.OFF}\n", file=sys.stderr)
    sys.exit(code)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------- contract

def load_target(slug):
    path = ROOT / "targets" / slug / "target.json"
    if not path.exists():
        available = sorted(p.name for p in (ROOT / "targets").iterdir() if p.is_dir())
        die(f"no target '{slug}'. Available: {', '.join(available) or '(none)'}")
    t = json.loads(path.read_text())
    t["_dir"] = path.parent
    t["_slug"] = slug
    return t


def prompt_set(target, held_out=False):
    """Load prompts in a fixed, sorted order.

    Prompts are literal files, never generated. The site's own harness
    cache-busts with a unique per-run prefix, which is right for throughput
    and fatal here: token identity needs the exact same bytes every time.
    """
    d = target["_dir"] / ("prompts-heldout" if held_out else "prompts")
    if not d.exists():
        if held_out:
            return []
        die(f"target {target['_slug']} has no prompts/ directory")
    out = []
    for p in sorted(d.glob("*.txt")):
        meta = target.get("promptSettings", {}).get(p.stem, {})
        out.append({
            "id": p.stem,
            "text": p.read_text(),
            "maxTokens": meta.get("maxTokens", target.get("defaultMaxTokens", 256)),
        })
    return out


# -------------------------------------------------------------------- thermal

def gpu_temp_c():
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()[0]
        return float(raw)
    except Exception:
        return None


def gpu_power_w():
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()[0]
        return float(raw)
    except Exception:
        return None


def thermal_gate():
    """Block until the part is cool enough that the arms are comparable.

    Spark-1 has been observed at 84 C under sustained load while Spark-2 idled
    at 43. Timing a paired run across that spread measures the room, not the
    kernel.
    """
    cfg = CONTRACT["measurement"]["thermalGate"]
    ceiling, budget = cfg["maxStartTempC"], cfg["maxWaitSeconds"]
    t = gpu_temp_c()
    if t is None:
        Out.warn("no nvidia-smi temperature -- thermal gate skipped, results are advisory")
        return None
    waited = 0
    while t > ceiling and waited < budget:
        Out.info(f"thermal gate: {t:.0f} C > {ceiling} C, waiting ({waited}s/{budget}s)")
        time.sleep(20)
        waited += 20
        t = gpu_temp_c()
    if t > ceiling:
        die(f"thermal gate never cleared: {t:.0f} C after {budget}s. "
            f"Something else is using the node, or the ceiling ({ceiling} C) is wrong for this room.")
    Out.info(f"thermal gate: {t:.0f} C (ceiling {ceiling} C)")
    return t


# ---------------------------------------------------------------------- build

def tree_fingerprint(tree: Path):
    """Hash the editable surface so we can skip a 15-minute rebuild."""
    h = hashlib.sha256()
    files = []
    for pattern in CONTRACT["editablePaths"]:
        base = pattern.split("**")[0].rstrip("/")
        root = tree / base
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(p for p in root.rglob("*") if p.is_file()))
    for f in sorted(set(files)):
        h.update(str(f.relative_to(tree)).encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()[:16]


def build(tree: Path, label):
    if not tree.exists():
        die(f"{tree} is missing -- run ./setup.sh first")
    stamp = tree / ".arena-build-fingerprint"
    fp = tree_fingerprint(tree)
    binary = tree / "build" / "bin" / "llama-server"
    if binary.exists() and stamp.exists() and stamp.read_text().strip() == fp:
        Out.info(f"{label}: build up to date ({fp})")
        return binary
    Out.step(f"building {label} ({fp}) -- ~15 min cold, seconds warm")
    flags = CONTRACT["vendor"]["buildFlags"]
    subprocess.run(["cmake", "-B", "build", *flags], cwd=tree, check=True)
    subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j"],
                   cwd=tree, check=True)
    if not binary.exists():
        die(f"{label}: build finished but {binary} does not exist")
    stamp.write_text(fp + "\n")
    return binary


# --------------------------------------------------------------------- server

class Server:
    """llama-server for one arm. Ready in ~3 s warm / ~14 s cold."""

    def __init__(self, binary: Path, target, log_path: Path):
        self.binary = binary
        self.target = target
        self.log_path = log_path
        self.proc = None
        self.port = target.get("port", 8080)
        self.base = f"http://127.0.0.1:{self.port}"

    def _argv(self):
        args = [str(self.binary)]
        for a in self.target["serveArgs"]:
            args.append(os.path.expanduser(a))
        args += ["--host", "127.0.0.1", "--port", str(self.port)]
        for flag in CONTRACT["measurement"]["serverFlags"]:
            if flag.split()[0] not in args:
                args += flag.split()
        return args

    def __enter__(self):
        argv = self._argv()
        Out.info("serve: " + " ".join(argv))
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            argv, stdout=self.log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + self.target.get("startupTimeout", 900)
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = self.log_path.read_text()[-2000:]
                die(f"llama-server exited during startup (rc={self.proc.returncode}):\n{tail}")
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=3) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            self.__exit__(None, None, None)
            die("llama-server never became healthy")
        self.props = self._props()
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            # kill the process group, by pid. Never pkill -f: over SSH your own
            # command line contains the pattern and you kill the session.
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        try:
            self.log.close()
        except Exception:
            pass
        return False

    def _props(self):
        try:
            with urllib.request.urlopen(f"{self.base}/props", timeout=10) as r:
                return json.loads(r.read())
        except Exception:
            return {}

    def n_ctx_slot(self):
        """Read the real per-slot context out of the log.

        In llama.cpp -c is the TOTAL context divided across slots, so this is
        the number that matters, not the -c you passed.
        """
        m = re.search(r"n_ctx_slot\s*=\s*(\d+)", self.log_path.read_text())
        return int(m.group(1)) if m else None

    def complete(self, prompt, max_tokens):
        """One greedy, streaming, batch-1 request. Returns text + timings."""
        body = {
            "model": self.target.get("modelName", "arena"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": self.target.get("seed", 0),
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.target.get("extraBody", {}),
        }
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks, usage = [], {}
        t0 = time.perf_counter()
        t_first = None
        with urllib.request.urlopen(req, timeout=self.target.get("requestTimeout", 1800)) as res:
            for raw in res:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                d = json.loads(payload)
                if d.get("usage"):
                    usage = d["usage"]
                for choice in d.get("choices", []):
                    piece = choice.get("delta", {}).get("content")
                    if piece:
                        if t_first is None:
                            t_first = time.perf_counter()
                        chunks.append(piece)
        t_end = time.perf_counter()
        text = "".join(chunks)
        if t_first is None:
            die("server returned no content. If this model does interleaved "
                "thinking, the text is in reasoning_content -- pass "
                "--reasoning-budget 0 in the target's serveArgs.")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens") or len(chunks)
        ttft = t_first - t0
        decode_s = t_end - t_first
        return {
            "text": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "ttftSeconds": round(ttft, 4),
            "decodeTps": round((completion_tokens - 1) / decode_s, 3) if decode_s > 0 and completion_tokens > 1 else None,
            "prefillTps": round(prompt_tokens / ttft, 2) if prompt_tokens and ttft > 0 else None,
        }


# ------------------------------------------------------------------------ arm

def run_arm(binary, target, prompts, label, tag):
    """One measured arm: thermal gate -> serve -> discard warmup -> measure."""
    Out.step(f"arm: {label}")
    temp_start = thermal_gate()
    logs = ROOT / "results" / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{target['_slug']}-{tag}.log"

    with Server(binary, target, log_path) as srv:
        n_ctx = srv.n_ctx_slot()
        if n_ctx is not None:
            Out.info(f"n_ctx_slot = {n_ctx}")
            want = target.get("expectContextTokens")
            if want and n_ctx != want:
                die(f"n_ctx_slot is {n_ctx}, target declares {want}. "
                    f"In llama.cpp -c is split across --parallel slots -- "
                    f"this is how you accidentally measure a quarter of the context you think you have.")

        # Warmup. The first request after server start differs on both engines;
        # keep it and every comparison reports a phantom mismatch.
        n_warm = CONTRACT["measurement"]["warmupRequests"]
        for i in range(n_warm):
            srv.complete("warmup", 32)
        Out.info(f"discarded {n_warm} warmup request(s)")

        results = []
        for p in prompts:
            r = srv.complete(p["text"], p["maxTokens"])
            r["id"] = p["id"]
            results.append(r)
            Out.info(f"{p['id']:<18} decode {r['decodeTps']:>8} tok/s   "
                     f"prefill {str(r['prefillTps']):>9} tok/s   "
                     f"{r['promptTokens']}->{r['completionTokens']} tok   {r['sha256'][:12]}")
        build_info = srv.props.get("build_info")

    return {
        "label": label,
        "at": now_iso(),
        "buildInfo": build_info,
        "tempStartC": temp_start,
        "tempEndC": gpu_temp_c(),
        "powerEndW": gpu_power_w(),
        "prompts": results,
        "decodeTps": statistics.median([r["decodeTps"] for r in results if r["decodeTps"]]),
        "prefillTps": statistics.median([r["prefillTps"] for r in results if r["prefillTps"]]),
    }


# ---------------------------------------------------------------- gate: diff

def vendor_diff_files():
    """Files the candidate tree changes relative to the pin."""
    pin = CONTRACT["vendor"]["commit"]
    try:
        tracked = subprocess.run(
            ["git", "diff", "--name-only", pin], cwd=CAND_TREE,
            capture_output=True, text=True, check=True).stdout.split()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=CAND_TREE,
            capture_output=True, text=True, check=True).stdout.split()
    except subprocess.CalledProcessError as e:
        die(f"could not diff vendor tree against pin {pin}: {e.stderr.strip()}")
    return sorted(set(tracked) | set(untracked))


def gate_editable_surface(target, verbose=True):
    changed = vendor_diff_files()
    patterns = CONTRACT["editablePaths"]
    illegal = []
    for f in changed:
        if not any(fnmatch.fnmatch(f, pat) or fnmatch.fnmatch(f, pat.replace("/**", "/*"))
                   or (pat.endswith("/**") and f.startswith(pat[:-3] + "/"))
                   for pat in patterns):
            illegal.append(f)
    if verbose:
        if not changed:
            Out.warn("vendor tree is identical to the pin -- nothing to measure yet")
        else:
            Out.info(f"{len(changed)} file(s) changed vs pin {CONTRACT['vendor']['commit']}")
            for f in changed:
                Out.info(f"  {f}")
    if illegal:
        for f in illegal:
            Out.fail(f"outside editablePaths: {f}")
        return False, changed
    if verbose and changed:
        Out.ok("gate 1 editable-surface")
    return True, changed


def gate_model_integrity(target):
    """The pinned-entry-count equivalent: same weights, or it is a recipe."""
    integ = target.get("modelIntegrity")
    if not integ:
        Out.warn("target declares no modelIntegrity -- skipping")
        return True
    path = Path(os.path.expanduser(integ["path"]))
    if not path.exists():
        die(f"model file missing: {path}")
    size = path.stat().st_size
    if size != integ["bytes"]:
        Out.fail(f"model size {size} != pinned {integ['bytes']}")
        return False
    if integ.get("sha256") and integ.get("verifySha", True):
        Out.info("hashing weights (20 GB, ~1 min)...")
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(8 << 20), b""):
                h.update(block)
        if h.hexdigest() != integ["sha256"]:
            Out.fail(f"model sha256 mismatch -- these are different weights, which makes this a recipe, not an optimization")
            return False
    Out.ok("model integrity")
    return True


# ---------------------------------------------------------------- gate: tokens

def gate_token_identity(arm, goldens, label="gate 2 token-identity"):
    if not goldens:
        Out.warn("no goldens recorded -- run `baseline` first")
        return False
    ok = True
    for r in arm["prompts"]:
        want = goldens.get(r["id"])
        if want is None:
            Out.fail(f"{r['id']}: no golden recorded")
            ok = False
        elif want["sha256"] != r["sha256"]:
            Out.fail(f"{r['id']}: output changed ({want['sha256'][:12]} -> {r['sha256'][:12]})")
            ok = False
    if ok:
        Out.ok(f"{label} ({len(arm['prompts'])}/{len(arm['prompts'])} identical)")
    return ok


# ---------------------------------------------------------------- commands

def cmd_baseline(args):
    target = load_target(args.target)
    prompts = prompt_set(target)
    Out.step(f"baseline: {target['_slug']} ({len(prompts)} prompts)")
    if not args.no_integrity:
        gate_model_integrity(target)
    binary = build(BASE_TREE, "base tree")

    arms = [run_arm(binary, target, prompts, "baseline", f"baseline-{i}") for i in range(args.repeats)]

    # Determinism self-check: the baseline arms must agree with each other
    # before they are allowed to define a golden.
    goldens = {}
    for r in arms[0]["prompts"]:
        goldens[r["id"]] = {"sha256": r["sha256"], "chars": len(r["text"]),
                            "completionTokens": r["completionTokens"]}
    for arm in arms[1:]:
        for r in arm["prompts"]:
            if goldens[r["id"]]["sha256"] != r["sha256"]:
                die(f"baseline is not deterministic on prompt '{r['id']}' across repeats. "
                    f"The token-identity gate cannot be trusted on this target -- check "
                    f"--parallel 1, temperature 0, and that warmup is being discarded.")
    if args.repeats > 1:
        Out.ok(f"baseline determinism ({args.repeats} arms agree on all prompts)")

    (target["_dir"] / "goldens.json").write_text(json.dumps({
        "contractVersion": CONTRACT["contractVersion"],
        "vendorCommit": CONTRACT["vendor"]["commit"],
        "recordedAt": now_iso(),
        "buildInfo": arms[0]["buildInfo"],
        "prompts": goldens,
    }, indent=2) + "\n")

    if args.keep_text:
        d = target["_dir"] / "goldens"
        d.mkdir(exist_ok=True)
        for r in arms[0]["prompts"]:
            (d / f"{r['id']}.txt").write_text(r["text"])

    (target["_dir"] / "baseline.json").write_text(json.dumps({
        "contractVersion": CONTRACT["contractVersion"],
        "vendorCommit": CONTRACT["vendor"]["commit"],
        "recordedAt": now_iso(),
        "node": os.uname().nodename,
        "buildInfo": arms[0]["buildInfo"],
        "decodeTps": statistics.median([a["decodeTps"] for a in arms]),
        "prefillTps": statistics.median([a["prefillTps"] for a in arms]),
        "arms": arms,
        "note": "Absolute numbers are for the record only. Scoring uses paired "
                "ratios measured in the same session -- a baseline recorded on "
                "another day is not comparable (24.05 -> 20.09 tok/s, observed).",
    }, indent=2) + "\n")
    Out.step(f"recorded goldens.json + baseline.json  "
             f"(decode {statistics.median([a['decodeTps'] for a in arms]):.2f} tok/s)")


def cmd_gate(args):
    target = load_target(args.target)
    ok, _ = gate_editable_surface(target)
    sys.exit(0 if ok else 1)


def cmd_bench(args):
    target = load_target(args.target)
    prompts = prompt_set(target)
    goldens_path = target["_dir"] / "goldens.json"
    goldens = json.loads(goldens_path.read_text())["prompts"] if goldens_path.exists() else {}

    Out.step(f"gate 1: editable surface")
    surface_ok, changed = gate_editable_surface(target)
    if not surface_ok and not args.force:
        die("submission touches paths outside editablePaths -- rejected at gate 1")
    if not args.no_integrity:
        gate_model_integrity(target)

    base_bin = build(BASE_TREE, "base tree")
    cand_bin = build(CAND_TREE, "candidate tree")

    pairs = args.pairs or CONTRACT["scoring"]["pairs"]
    ratios, arms = [], []
    for i in range(pairs):
        b = run_arm(base_bin, target, prompts, f"baseline (pair {i+1}/{pairs})", f"pair{i}-base")
        c = run_arm(cand_bin, target, prompts, f"candidate (pair {i+1}/{pairs})", f"pair{i}-cand")
        arms += [b, c]
        ratios.append({
            "decode": c["decodeTps"] / b["decodeTps"],
            "prefill": c["prefillTps"] / b["prefillTps"],
        })
        Out.info(f"pair {i+1}: decode x{ratios[-1]['decode']:.4f}  prefill x{ratios[-1]['prefill']:.4f}")

    decode_speedup = statistics.median(r["decode"] for r in ratios)
    prefill_speedup = statistics.median(r["prefill"] for r in ratios)

    print()
    Out.step("gate 2: token identity (candidate vs goldens)")
    identity_ok = all(gate_token_identity(a, goldens) for a in arms if a["label"].startswith("candidate"))

    Out.step("gate 4: speedup floors")
    floor = CONTRACT["scoring"]["floor"]
    floors_ok = decode_speedup >= floor and prefill_speedup >= floor
    for name, val in (("decode", decode_speedup), ("prefill", prefill_speedup)):
        (Out.ok if val >= floor else Out.fail)(f"{name} speedup x{val:.4f} (floor {floor})")

    score = (decode_speedup ** CONTRACT["scoring"]["decodeExponent"]) * \
            (prefill_speedup ** CONTRACT["scoring"]["prefillExponent"])

    record = {
        "contractVersion": CONTRACT["contractVersion"],
        "target": target["_slug"],
        "at": now_iso(),
        "node": os.uname().nodename,
        "vendorCommit": CONTRACT["vendor"]["commit"],
        "changedFiles": changed,
        "pairs": pairs,
        "ratios": ratios,
        "decodeSpeedup": round(decode_speedup, 5),
        "prefillSpeedup": round(prefill_speedup, 5),
        "score": round(score, 5),
        "gates": {
            "editableSurface": surface_ok,
            "tokenIdentity": identity_ok,
            "speedupFloors": floors_ok,
        },
        "promotable": bool(surface_ok and identity_ok and floors_ok and score > 1.0),
        "arms": arms,
    }
    out = ROOT / "results" / f"{target['_slug']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")

    print()
    verdict = f"{Out.GRN}SCORE {score:.4f}{Out.OFF}" if record["promotable"] else f"{Out.RED}REJECTED{Out.OFF}"
    print(f"{Out.BOLD}{target['_slug']}{Out.OFF}: decode x{decode_speedup:.4f}  "
          f"prefill x{prefill_speedup:.4f}  ->  {verdict}")
    print(f"    written to {out.relative_to(ROOT)}")
    if record["promotable"]:
        print(f"    {Out.DIM}gate 3 (held-out prompts) runs on the referee's node, not here.{Out.OFF}")
    sys.exit(0 if record["promotable"] else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="record goldens + the baseline record for a target")
    b.add_argument("--target", required=True)
    b.add_argument("--repeats", type=int, default=2, help="baseline arms; >1 also proves determinism")
    b.add_argument("--keep-text", action="store_true", help="also write full golden text (large)")
    b.add_argument("--no-integrity", action="store_true")
    b.set_defaults(fn=cmd_baseline)

    n = sub.add_parser("bench", help="paired candidate-vs-baseline run with gates")
    n.add_argument("--target", required=True)
    n.add_argument("--pairs", type=int, default=None)
    n.add_argument("--force", action="store_true", help="measure even if gate 1 fails (never promotes)")
    n.add_argument("--no-integrity", action="store_true")
    n.set_defaults(fn=cmd_bench)

    g = sub.add_parser("gate", help="editable-surface check only, no measurement")
    g.add_argument("--target", required=True)
    g.set_defaults(fn=cmd_gate)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
