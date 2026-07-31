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


# Gate 3 used to compare tokens only; it now also times the arms. Old records
# carry the old id and are still valid verifications of what they checked.
HELD_OUT_GATE_IDS = ("held-out-identity-and-speedup", "held-out-token-identity")


def held_out_tolerance(default=0.5):
    """How much of a claimed gain must survive unseen prompts. From the contract,
    so the number is published rather than buried in the harness."""
    for g in CONTRACT.get("gates", []):
        if g.get("id") in HELD_OUT_GATE_IDS and "tolerance" in g:
            return g["tolerance"]
    return default


def require_local_node(record, path, kind, force=False):
    """Refuse a record measured somewhere else.

    Every number this harness publishes is a paired ratio, and a paired ratio is
    only meaningful on the node that produced it. `promote` copies the ratio
    straight out of the record it is handed, so handing it the contributor's
    record publishes the contributor's measurement under the referee's name. It
    would pass every other check in `promote`, and the site would then label it
    verified. Cheap to check, so check it.
    """
    node = record.get("node")
    here = os.uname().nodename
    if not node or node == here:
        return
    msg = (f"{kind} {path} was measured on '{node}', but this node is '{here}'.\n"
           f"    Gate 5 is the referee's, and so is the measurement behind it: re-run the\n"
           f"    submission here and promote YOUR record, not the contributor's.\n"
           f"      ./bench.sh --target <target>\n"
           f"    Pass --force only if you mean to publish a number you did not measure.")
    if force:
        Out.warn(f"--force: promoting a {kind} measured on '{node}', not '{here}'")
        return
    die(msg)


# ------------------------------------------------------------------- contract

def load_target(slug):
    path = ROOT / "targets" / slug / "target.json"
    if not path.exists():
        available = sorted(p.name for p in (ROOT / "targets").iterdir() if p.is_dir())
        die(f"no target '{slug}'. Available: {', '.join(available) or '(none)'}")
    t = json.loads(path.read_text())
    t["_dir"] = path.parent
    t["_slug"] = slug

    # Hard rule: a target must point at weights a contributor can DOWNLOAD.
    # Requiring someone to produce their own quantization puts hours of work and
    # ~88 GB of transient disk in front of their first line of CUDA, and most
    # people stop there. A barrier in front of the arena costs more than any
    # single kernel win inside it.
    w = t.get("weights")
    if not w or not w.get("hfRepo"):
        die(f"target '{slug}' declares no downloadable weights.\n"
            f"    Add a weights block naming a PUBLIC Hugging Face repo (and hfFile for a\n"
            f"    GGUF), verified not gated and not private. See policy.publicWeights.")
    if not w.get("public", False):
        Out.warn(f"{slug}: weights are NOT publicly downloadable "
                 f"({w.get('publicTodo', 'no plan recorded')})")
        Out.warn("  contributors cannot run this target without building the weights "
                 "themselves — it must not be anyone's entry point")
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


_IDLE_FLOOR = None


def idle_floor_c(samples=6, gap=5):
    """The coolest this node gets right now, measured once per run.

    The gate used to compare against a fixed 46 C, taken from an idle reading of
    43 C. That number does not hold: the same node, verifiably idle (load 0.04,
    105 GB free, no processes), has been observed sitting at 52 C with
    nvidia-smi also reporting 96% utilisation — which it is not. Its
    unified-memory readings are known to be unreliable, and an absolute ceiling
    below the current floor deadlocks the gate forever.

    So calibrate against the machine instead of against a remembered number. The
    gate's job is only that every arm STARTS from the same thermal state; that
    works off a floor + tolerance and does not care what the absolute value is.
    """
    global _IDLE_FLOOR
    if _IDLE_FLOOR is not None:
        return _IDLE_FLOOR
    readings = []
    for _ in range(samples):
        t = gpu_temp_c()
        if t is not None:
            readings.append(t)
        time.sleep(gap)
    _IDLE_FLOOR = min(readings) if readings else None
    if _IDLE_FLOOR is not None:
        Out.info(f"idle floor calibrated: {_IDLE_FLOOR:.0f} C "
                 f"(from {len(readings)} samples over {samples * gap}s)")
    return _IDLE_FLOOR


def thermal_gate():
    """Block until the part is cool enough that the arms are comparable.

    Spark-1 has been observed at 84 C under sustained load while Spark-2 idled
    at 43. Timing a paired run across that spread measures the room, not the
    kernel.
    """
    cfg = CONTRACT["measurement"]["thermalGate"]
    budget = cfg["maxWaitSeconds"]
    tolerance = cfg.get("toleranceC", 3)
    floor = idle_floor_c()
    ceiling = (floor + tolerance) if floor is not None else cfg["maxStartTempC"]
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
        die(f"thermal gate never cleared: {t:.0f} C after {budget}s "
            f"(floor {floor} C + {tolerance} C tolerance). Something else is using the node.")
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


def ensure_cuda_on_path():
    """DGX OS keeps nvcc in /usr/local/cuda/bin, which non-interactive shells
    (every `ssh spark python3 harness/arena.py ...`) do not have on PATH."""
    if shutil.which("nvcc"):
        return
    cuda_bin = "/usr/local/cuda/bin"
    if Path(cuda_bin, "nvcc").exists():
        os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"


def build(tree: Path, label):
    if not tree.exists():
        die(f"{tree} is missing -- run ./setup.sh first")
    ensure_cuda_on_path()
    # Outside the vendor tree on purpose: anything the harness writes in there
    # shows up as an untracked file and trips gate 1 on the submitter's own
    # diff. (Found by gate 1, on the harness itself.)
    stamp = ROOT / "results" / "_build" / f"{tree.name}.fingerprint"
    stamp.parent.mkdir(parents=True, exist_ok=True)
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


# -------------------------------------------------------------------- sandbox

_EGRESS_PROBE = (
    "import socket\n"
    "try:\n"
    "    socket.create_connection(('1.1.1.1', 443), timeout=4).close()\n"
    "    print('EGRESS_OPEN')\n"
    "except Exception:\n"
    "    print('EGRESS_BLOCKED')\n"
)


def sandbox_cfg():
    return CONTRACT.get("sandbox") or {}


def _sh(cmd, timeout=90):
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "").strip()


def sandbox_status():
    """Actively try to reach the internet as the sandbox user.

    An attempt, not an inspection of the ruleset: a rule that exists and does not
    match is exactly what this has to catch.
    """
    cfg = sandbox_cfg()
    user = cfg.get("user")
    if not user:
        return False, "contract declares no sandbox.user"
    if _sh(f"id -u {user} >/dev/null 2>&1 && echo yes || echo no") != "yes":
        return False, f"user '{user}' does not exist on this node"
    if _sh(f"sudo -n -u {user} true >/dev/null 2>&1 && echo yes || echo no") != "yes":
        return False, (f"cannot run as '{user}' without a password. See sandbox.setup "
                       f"in benchmark.json.")
    probe = Path("/tmp/arena_egress_probe.py")
    probe.write_text(_EGRESS_PROBE)
    out = _sh(f"sudo -n -u {user} python3 {probe} 2>&1 | tail -1")
    if "EGRESS_BLOCKED" in out:
        return True, f"egress from '{user}' is blocked"
    if "EGRESS_OPEN" in out:
        return False, (f"egress from '{user}' is OPEN. The REJECT rule is missing or not matching; "
                       f"a submission could fetch help mid-run.")
    return False, f"probe gave no verdict: {out[:120]}"


def cmd_sandbox(args):
    cfg = sandbox_cfg()
    Out.step(f"sandbox: mechanism '{cfg.get('mechanism')}', user '{cfg.get('user')}'")
    ok, detail = sandbox_status()
    (Out.ok if ok else Out.fail)(detail)
    if not ok:
        print()
        Out.info("one-time root setup (shared with the vLLM arena):")
        for line in cfg.get("setup", []):
            print(f"      {line}")
        print()
        Out.info(cfg.get("setupNote", ""))
    else:
        user = cfg["user"]
        for label, path in (("candidate binary dir", str(CAND_TREE / "build" / "bin")),
                            ("arena tree", str(ROOT))):
            r = _sh(f"sudo -n -u {user} test -r {path} && echo yes || echo no")
            (Out.ok if r == "yes" else Out.warn)(f"{label} readable by {user}: {r}")
    return 0 if ok else 1


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

    def _sandbox_guard(self):
        """No CANDIDATE-tree binary runs without verified network isolation.

        On the server rather than in run_arm: run_arm is one of several callers,
        and a guard that one path bypasses is not a guard. Baseline arms are
        exempt -- that tree is the pin.
        """
        if not sandbox_cfg().get("required"):
            return
        try:
            is_candidate = str(self.binary).startswith(str(CAND_TREE))
        except Exception:
            is_candidate = True          # unsure means treat it as untrusted
        if not is_candidate:
            return
        ok, detail = sandbox_status()
        if not ok:
            die("refusing to start a server built from the CANDIDATE tree without "
                "network isolation.\n"
                f"    {detail}\n"
                "    Gate 3 stops a submission that memorised the held-out answers. It cannot\n"
                "    stop one that fetches help while the arm runs. Run\n"
                "    `python3 harness/arena.py sandbox --check` for the one-time setup, or set\n"
                "    sandbox.required=false in benchmark.json and accept that submissions are\n"
                "    trusted.")
        Out.ok(f"sandbox: {detail}")

    def __enter__(self):
        self._sandbox_guard()
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
        print(f"    {Out.DIM}gate 3 (held-out identity AND speedup) runs on the referee's "
              f"node — pass --claimed-speedup {decode_speedup:.5f}.{Out.OFF}")
    sys.exit(0 if record["promotable"] else 1)



# ------------------------------------------------------------ held-out prompts

# Gate 3 needs prompts the submitter has never seen, on a PUBLIC repo. Storing
# them is the obvious approach and the wrong one: a file in the repo is visible,
# a gitignored file on the referee's node is a static set that leaks a little
# with every verification, and either way there is nothing to show a sceptic
# afterwards.
#
# So they are not stored at all. They are GENERATED from a random seed at
# verification time, and the seed is recorded in the result. Nobody — including
# the referee — can know the prompts in advance, and anybody can regenerate them
# afterwards from the seed to audit a disputed verification. Unknowable ahead,
# reproducible behind.
#
# Content is deliberately varied in shape and length rather than random noise:
# prompt length picks the flash-attention tile and the prefill path, and code vs
# prose changes which experts a MoE routes to. A held-out set that is all one
# shape gates one code path.

_HELDOUT_TOPICS = [
    "a ring buffer with a lock-free single-producer path",
    "a tokenizer that merges byte pairs from a frozen vocabulary",
    "a scheduler that admits requests under a fixed memory budget",
    "a parser for a small configuration language with includes",
    "a cache with time-to-live eviction and hit-rate accounting",
    "a diff algorithm over lines with a configurable context window",
    "a rate limiter using a sliding window over timestamps",
    "a binary format reader that validates a header before mmap",
]

_HELDOUT_SUBJECTS = [
    "why memory bandwidth, not FLOPs, sets decode speed on a unified-memory part",
    "how a mixture-of-experts model routes a token and what that costs in reads",
    "what a KV cache actually stores, and why sliding-window attention shrinks it",
    "why the first request after a server start is not comparable to the rest",
    "how quantization scope, not bit width alone, decides a model's decode rate",
    "what a speculative draft has to get right for acceptance to stay high",
    "why greedy decoding can still be non-deterministic once a batch forms",
    "how thermal drift corrupts an unpaired throughput comparison",
]


def _heldout_rand(seed_hex, *parts):
    """Deterministic 64-bit stream from the seed and a label. Stdlib only, and
    stable across Python versions -- hash() is not."""
    h = hashlib.sha256(seed_hex.encode())
    for p in parts:
        h.update(b"\x00")
        h.update(str(p).encode())
    return int.from_bytes(h.digest()[:8], "big")


def generate_heldout_prompts(target, seed_hex, count):
    """`count` prompts, fully determined by (seed_hex, target slug, index)."""
    prompts = []
    for i in range(count):
        r = lambda *k: _heldout_rand(seed_hex, target["_slug"], i, *k)
        # Spread lengths across the shapes the target actually serves: a short
        # interactive prompt, a couple of KB, and something long enough to reach
        # a different attention tile.
        length_class = i % 3
        n_filler = (0, 12, 90)[length_class] + r("filler") % 8
        if r("kind") % 2 == 0:
            topic = _HELDOUT_TOPICS[r("topic") % len(_HELDOUT_TOPICS)]
            task = (f"Write a complete, working Python implementation of {topic}. "
                    f"Include type hints, docstrings, and unittest test cases. "
                    f"Output only code.")
        else:
            subject = _HELDOUT_SUBJECTS[r("subject") % len(_HELDOUT_SUBJECTS)]
            task = (f"Explain in careful technical prose: {subject}. "
                    f"Work through the arithmetic where it matters and state "
                    f"what you would measure to check the claim.")
        filler = "".join(
            f"Note {j} (ref {r('ref', j) % 100000:05d}): "
            f"{_HELDOUT_SUBJECTS[r('fs', j) % len(_HELDOUT_SUBJECTS)]}. "
            for j in range(n_filler)
        )
        text = (f"{filler}\n\nIgnore the notes above.\n\n{task}" if filler else task)
        prompts.append({
            "id": f"heldout-{i:02d}",
            "text": text,
            # Longer than the public set: every extra token is another chance
            # for a reassociated kernel to fall off a different argmax.
            "maxTokens": target.get("heldOutMaxTokens", 384),
        })
    return prompts


def cmd_heldout(args):
    """Gate 3: paired token identity AND speedup, on prompts nobody has seen.

    Runs base and candidate back to back on freshly generated prompts, compares
    output hashes, and times both arms. No goldens are stored or needed -- the
    base arm computes them in the same session, which is also what makes them
    impossible to tune against.

    The arms were always timed; the numbers were simply thrown away. Keeping them
    turns "it does not go faster on my node" from the referee's opinion into a
    gate result, recorded next to the seed that produced it. The vLLM arena needs
    the same check to catch a memoizing patch. Here the editable surface is CUDA,
    so nothing can memoize -- what this catches instead is a win that lives only
    on the submitter's node, or only on the prompts they tuned against.
    """
    target = load_target(args.target)
    seed = args.seed or os.urandom(16).hex()
    prompts = generate_heldout_prompts(target, seed, args.count)

    Out.step(f"gate 3: held-out identity + speedup ({args.count} generated prompts)")
    Out.info(f"seed {seed}")
    Out.info("the seed is recorded in the result, so these prompts can be "
             "regenerated to audit this verification -- but not before it")

    base_bin = build(BASE_TREE, "base tree")
    cand_bin = build(CAND_TREE, "candidate tree")

    # One pair by default: this gate already costs ~11 minutes, and the pair is
    # what makes the ratio a ratio. More pairs tighten the median when a claim
    # is close to the tolerance.
    pairs = max(1, args.pairs or 1)
    ratios, arm_pairs = [], []
    for i in range(pairs):
        label = f" (pair {i+1}/{pairs})" if pairs > 1 else ""
        b = run_arm(base_bin, target, prompts, f"held-out baseline{label}", f"heldout{i}-base")
        c = run_arm(cand_bin, target, prompts, f"held-out candidate{label}", f"heldout{i}-cand")
        arm_pairs.append((b, c))
        ratios.append({
            "decode": c["decodeTps"] / b["decodeTps"],
            "prefill": c["prefillTps"] / b["prefillTps"],
        })
        if pairs > 1:
            Out.info(f"pair {i+1}: decode x{ratios[-1]['decode']:.4f}  "
                     f"prefill x{ratios[-1]['prefill']:.4f}")

    held_decode = statistics.median(r["decode"] for r in ratios)
    held_prefill = statistics.median(r["prefill"] for r in ratios)

    mismatches = []
    for base_arm, cand_arm in arm_pairs:
        for b, c in zip(base_arm["prompts"], cand_arm["prompts"]):
            if b["sha256"] != c["sha256"]:
                mismatches.append({
                    "id": b["id"],
                    "baseSha256": b["sha256"],
                    "candidateSha256": c["sha256"],
                    "baseTokens": b["completionTokens"],
                    "candidateTokens": c["completionTokens"],
                })
                Out.fail(f"{b['id']}: output differs ({b['sha256'][:12]} -> {c['sha256'][:12]})")
    identical = not mismatches
    if identical:
        Out.ok(f"held-out identity ({len(prompts)}/{len(prompts)} identical on unseen prompts)")

    # The claimed gain has to survive prompts the submitter never saw. Stated as
    # a fraction of the EXCESS over 1.0, not of the ratio: x1.02 claimed against
    # x1.01 measured is half the win, and comparing raw ratios would call that a
    # 99% match.
    tol = held_out_tolerance()
    claimed = args.claimed_speedup
    generalizes = None
    if claimed and claimed > 1.0:
        need = 1.0 + (claimed - 1.0) * tol
        generalizes = held_decode >= need
        (Out.ok if generalizes else Out.fail)(
            f"held-out decode x{held_decode:.4f} vs claimed x{claimed:.4f} "
            f"(needs >= x{need:.4f}, {tol:.0%} of the claimed gain)")
        if not generalizes:
            Out.info("a win that does not reproduce on unseen prompts, on this node, is "
                     "not a win this frontier can carry")
    else:
        Out.warn("no --claimed-speedup given; speed generalization NOT checked. Pass the "
                 "decodeSpeedup from the bench record you are refereeing.")

    passed = identical and generalizes is not False

    record = {
        "contractVersion": CONTRACT["contractVersion"],
        "gate": "held-out-identity-and-speedup",
        "target": target["_slug"],
        "at": now_iso(),
        "node": os.uname().nodename,
        "referee": args.referee,
        "vendorCommit": CONTRACT["vendor"]["commit"],
        "changedFiles": vendor_diff_files(),
        "seed": seed,
        "promptCount": args.count,
        "promptIds": [p["id"] for p in prompts],
        "pairs": pairs,
        "ratios": ratios,
        "heldOutDecodeSpeedup": round(held_decode, 5),
        "heldOutPrefillSpeedup": round(held_prefill, 5),
        "claimedDecodeSpeedup": claimed,
        "tolerance": tol,
        "identical": identical,
        "generalizes": generalizes,
        "passed": passed,
        "mismatches": mismatches,
        # Everything an auditor needs to land on the same verdict, not just the
        # same prompts: the seed, the count, the pair count and the claim.
        "regenerate": (f"python3 harness/arena.py heldout --target {target['_slug']} "
                       f"--seed {seed} --count {args.count} --pairs {pairs}"
                       + (f" --claimed-speedup {claimed}" if claimed else "")),
        "note": "Prompts are generated from the seed, never stored. Regenerating "
                "with the same seed reproduces them exactly; without the record "
                "they cannot be known in advance.",
    }
    out = ROOT / "results" / target["_slug"] / f"heldout-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    Out.info(f"written to {out.relative_to(ROOT)}")
    sys.exit(0 if passed else 1)


# ------------------------------------------------------------- leaderboard

def leaderboard_path(target):
    return ROOT / "results" / target["_slug"] / "leaderboard.json"


def load_leaderboard(target, create=False):
    """The ordered promotion chain for one target.

    This file, not results/*.json, is what a frontier chart is drawn from. A
    bench record says "x1.04 against whatever the incumbent was that day";
    only an ordered chain can say "x1.31 against the original baseline", and
    that chain cannot be reconstructed after the fact -- which is why it is
    written from the first promotion rather than added once there is something
    to plot.
    """
    path = leaderboard_path(target)
    if path.exists():
        return json.loads(path.read_text())
    if not create:
        die(f"no leaderboard for {target['_slug']} -- run `promote` on a passing record first")
    base_path = target["_dir"] / "baseline.json"
    if not base_path.exists():
        die(f"no baseline.json for {target['_slug']} -- run `baseline` first")
    base = json.loads(base_path.read_text())
    return {
        "contractVersion": CONTRACT["contractVersion"],
        "target": target["_slug"],
        "targetName": target.get("name", target["_slug"]),
        "recipe": target.get("recipe"),
        "engine": CONTRACT["engine"],
        "hardware": CONTRACT["hardware"]["name"],
        "vendorPin": CONTRACT["vendor"]["commit"],
        "repoUrl": CONTRACT.get("repoUrl"),
        "origin": {
            "recordedAt": base["recordedAt"],
            "node": base["node"],
            "buildInfo": base["buildInfo"],
            "decodeTps": base["decodeTps"],
            "prefillTps": base["prefillTps"],
            "note": "The frontier is measured against THIS. Every cumulative "
                    "number on the chart is a product of paired ratios back to here.",
        },
        "current": {
            "decodeTps": base["decodeTps"],
            "prefillTps": base["prefillTps"],
            "cumulativeDecodeSpeedup": 1.0,
            "cumulativePrefillSpeedup": 1.0,
            "cumulativeScore": 1.0,
            "promotionCount": 0,
        },
        "promotions": [],
    }


def cmd_promote(args):
    """Promote a passing bench record onto the frontier.

    Deliberately a separate step from `bench`. Gate 3 (held-out prompts) and
    gate 5 (beat the incumbent) are the referee's, not the contributor's, so
    the thing that writes history is run by whoever ran those.
    """
    target = load_target(args.target)
    record = json.loads(Path(args.record).read_text())

    if record["target"] != target["_slug"]:
        die(f"record is for target '{record['target']}', not '{target['_slug']}'")
    if record["contractVersion"] != CONTRACT["contractVersion"]:
        die(f"record is contractVersion {record['contractVersion']}, contract is "
            f"{CONTRACT['contractVersion']} -- scores are only comparable within one version")
    if not record.get("promotable") and not args.force:
        failed = [k for k, v in record["gates"].items() if not v]
        die(f"record did not pass its own gates ({', '.join(failed) or 'score <= 1.0'})")
    # The ratio published to the frontier is the one in THIS record, so the
    # record has to be the referee's own re-run. Promoting the contributor's
    # record is the easy mistake: it passes every other check here, and the
    # number on the chart then belongs to a node nobody else measured -- while
    # the site presents it as verified. Gate 5 is the referee's; so is the
    # measurement it rests on.
    require_local_node(record, args.record, "bench record", args.force)
    # Gate 3 must be BACKED BY A RECORD, not asserted. A boolean flag is a
    # promise; a record names the seed, the node and the changed files, so the
    # verification can be reproduced by anyone later. "Verified" is granted, not
    # owed -- and a claim nobody can re-check is not a grant.
    held_out = None
    if args.held_out_record:
        held_out = json.loads(Path(args.held_out_record).read_text())
        # Records written before gate 3 was timed carry the older id. Both are
        # real verifications, so both are accepted; only the newer one can also
        # report whether the win generalized.
        if held_out.get("gate") not in HELD_OUT_GATE_IDS:
            die(f"{args.held_out_record} is not a held-out verification record")
        if held_out["target"] != target["_slug"]:
            die(f"held-out record is for target '{held_out['target']}'")
        require_local_node(held_out, args.held_out_record, "held-out record", args.force)
        # Two halves, two messages. Gate 3 can now fail for a reason that has
        # nothing to do with tokens, and reporting a speed failure as "changes
        # output" would send the contributor after the wrong bug.
        if held_out.get("mismatches") or held_out.get("identical") is False:
            die(f"held-out verification FAILED on {len(held_out.get('mismatches', []))} "
                f"prompt(s) — this candidate changes output on inputs it was not "
                f"tuned against, which is exactly what gate 3 exists to catch.")
        # The record has to describe THIS candidate. Verifying one diff and
        # promoting another is the obvious way to launder a failing submission.
        if sorted(held_out.get("changedFiles", [])) != sorted(record.get("changedFiles", [])):
            die("held-out record was produced against a different diff than the "
                "bench record.\n    heldout : "
                f"{sorted(held_out.get('changedFiles', []))}\n    bench   : "
                f"{sorted(record.get('changedFiles', []))}\n"
                "    Re-run both against the same tree.")
        # The timed half of gate 3. A kernel cannot memoize a completion, so
        # this arena does not need the check to stop cheating the way the vLLM
        # arena does -- it needs it to stop publishing a win that only exists on
        # the contributor's node, or on the prompts they tuned against.
        if held_out.get("generalizes") is False:
            die(f"held-out decode x{held_out.get('heldOutDecodeSpeedup')} did not reproduce "
                f"the claimed x{held_out.get('claimedDecodeSpeedup')} on unseen prompts "
                f"(needed {held_out.get('tolerance', 0.5):.0%} of the claimed gain). "
                "A speedup that does not survive inputs the submitter never saw is not "
                "one this frontier can carry.")
        if held_out.get("generalizes") is None:
            Out.warn("held-out record has no speed check (no --claimed-speedup was passed). "
                     "Gate 3 confirmed identity only; the speedup on the chart rests on the "
                     "bench record alone.")
        # Catch-all: a record that failed for a reason this version does not know
        # about is still a failed record.
        if not held_out.get("passed"):
            die(f"{args.held_out_record} did not pass gate 3.")
    elif not args.force:
        die("gate 3 (held-out identity and speedup) has no verification record.\n"
            "    Run it on the referee's node:\n"
            f"      python3 harness/arena.py heldout --target {target['_slug']} "
            f"--claimed-speedup {record.get('decodeSpeedup')}\n"
            "    then pass --held-out-record results/<target>/heldout-<stamp>.json.\n"
            "    This is the gate that catches a kernel which reassociated its way "
            "into a different argmax on everything except the prompts it was tuned "
            "against -- promoting without it is the one shortcut that silently "
            "corrupts the corpus.")

    lb = load_leaderboard(target, create=True)
    seq = len(lb["promotions"]) + 1

    # Cumulative = product of paired ratios back to the origin. Each ratio was
    # measured against the incumbent of its day, so the chain multiplies. It
    # never re-derives from absolute tok/s: absolutes are not comparable across
    # days on this fleet (24.05 -> 20.09 observed overnight).
    prev = lb["current"]
    cum_decode = prev["cumulativeDecodeSpeedup"] * record["decodeSpeedup"]
    cum_prefill = prev["cumulativePrefillSpeedup"] * record["prefillSpeedup"]
    cum_score = (cum_decode ** CONTRACT["scoring"]["decodeExponent"]) * \
                (cum_prefill ** CONTRACT["scoring"]["prefillExponent"])

    promotion = {
        "seq": seq,
        "promotedAt": now_iso(),
        "measuredAt": record["at"],
        "author": {
            "handle": args.author,
            "name": args.author_name or args.author,
            "model": args.model,
        },
        "note": Path(args.note).read_text().strip() if args.note and Path(args.note).exists() else (args.note or ""),
        "submissionUrl": args.url,
        "scope": args.scope,
        "alsoMoved": args.also_moved or [],
        "changedFiles": record.get("changedFiles", []),
        "kernels": sorted({Path(f).name for f in record.get("changedFiles", [])
                           if f.startswith("ggml/src/ggml-cuda/")}),
        "vsIncumbent": {
            "decodeSpeedup": record["decodeSpeedup"],
            "prefillSpeedup": record["prefillSpeedup"],
            "score": record["score"],
            "pairs": record["pairs"],
        },
        "cumulative": {
            "decodeSpeedup": round(cum_decode, 5),
            "prefillSpeedup": round(cum_prefill, 5),
            "score": round(cum_score, 5),
            "percentFaster": round((cum_decode - 1) * 100, 2),
            # Absolutes projected from the origin through the ratio chain, so a
            # leaderboard row can show tok/s without anyone re-measuring an old
            # submission on today's thermals.
            "decodeTps": round(lb["origin"]["decodeTps"] * cum_decode, 3),
            "prefillTps": round(lb["origin"]["prefillTps"] * cum_prefill, 2),
            # What THIS solver added, in percentage points of the headline. Not
            # the same as (decodeSpeedup - 1): a x1.01 win on top of a x1.37
            # frontier adds 1.4pp, not 1.0pp. This is the number the leaderboard
            # shows in green, and getting it from the ratio would understate
            # every late win.
            "addedPct": round((cum_decode - prev["cumulativeDecodeSpeedup"]) * 100, 2),
        },
        "verification": {
            "node": record["node"],
            "gates": record["gates"],
            "heldOutVerified": bool(held_out and held_out.get("passed")),
            "heldOut": {
                "seed": held_out["seed"],
                "promptCount": held_out["promptCount"],
                "at": held_out["at"],
                "node": held_out["node"],
                "regenerate": held_out["regenerate"],
            } if held_out else None,
            "referee": args.referee,
        },
    }
    lb["promotions"].append(promotion)
    lb["current"] = {
        # Absolute tok/s is projected from the origin through the ratio chain,
        # not read off today's run, so the series stays internally consistent.
        "decodeTps": round(lb["origin"]["decodeTps"] * cum_decode, 3),
        "prefillTps": round(lb["origin"]["prefillTps"] * cum_prefill, 2),
        "cumulativeDecodeSpeedup": round(cum_decode, 5),
        "cumulativePrefillSpeedup": round(cum_prefill, 5),
        "cumulativeScore": round(cum_score, 5),
        "promotionCount": seq,
        "leaderHandle": args.author,
        "updatedAt": now_iso(),
    }

    path = leaderboard_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lb, indent=2) + "\n")
    Out.step(f"promoted #{seq} by {args.author}")
    Out.info(f"vs incumbent : decode x{record['decodeSpeedup']:.4f}  prefill x{record['prefillSpeedup']:.4f}")
    Out.info(f"cumulative   : decode x{cum_decode:.4f}  ({(cum_decode-1)*100:+.2f}% vs origin)")
    Out.info(f"frontier     : {lb['current']['decodeTps']} tok/s decode")
    Out.info(f"written to   : {path.relative_to(ROOT)}")


def cmd_leaderboard(args):
    target = load_target(args.target)
    lb = load_leaderboard(target, create=args.init)
    if args.init and not leaderboard_path(target).exists():
        # A target with zero promotions still publishes a leaderboard: the site
        # needs the origin to draw the flat baseline the frontier departs from,
        # and an empty chart is a truthful chart.
        leaderboard_path(target).parent.mkdir(parents=True, exist_ok=True)
        leaderboard_path(target).write_text(json.dumps(lb, indent=2) + "\n")
        Out.info(f"initialized {leaderboard_path(target).relative_to(ROOT)}")
    cur = lb["current"]
    print(f"\n{Out.BOLD}{lb['targetName']}{Out.OFF}  ({lb['target']})")
    print(f"  recipe    : {lb.get('recipe') or '-'}")
    print(f"  origin    : {lb['origin']['decodeTps']} tok/s decode, "
          f"{lb['origin']['prefillTps']} tok/s prefill  ({lb['origin']['recordedAt'][:10]})")
    print(f"  frontier  : {cur['decodeTps']} tok/s decode  "
          f"{Out.GRN}{(cur['cumulativeDecodeSpeedup']-1)*100:+.2f}%{Out.OFF} "
          f"over {cur['promotionCount']} promotion(s)\n")
    if not lb["promotions"]:
        print(f"  {Out.DIM}no promotions yet -- the frontier is the baseline{Out.OFF}\n")
        return
    print(f"  {'#':>2}  {'when':<11} {'who':<18} {'vs inc':>8} {'cumulative':>11}  kernels")
    for p in lb["promotions"]:
        print(f"  {p['seq']:>2}  {p['promotedAt'][:10]:<11} {p['author']['handle'][:18]:<18} "
              f"x{p['vsIncumbent']['decodeSpeedup']:.4f} {p['cumulative']['percentFaster']:>+10.2f}%  "
              f"{', '.join(p['kernels'][:3]) or '-'}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="record goldens + the baseline record for a target")
    b.add_argument("--target", required=True)
    b.add_argument("--repeats", type=int, default=2, help="baseline arms; >1 also proves determinism")
    b.add_argument("--keep-text", action="store_true", help="also write full golden text (large)")
    b.add_argument("--no-integrity", action="store_true")
    b.set_defaults(fn=cmd_baseline)

    sb = sub.add_parser("sandbox", help="check that isolation for candidate arms actually works")

    sb.add_argument("--check", action="store_true", default=True)

    sb.set_defaults(fn=cmd_sandbox)


    n = sub.add_parser("bench", help="paired candidate-vs-baseline run with gates")
    n.add_argument("--target", required=True)
    n.add_argument("--pairs", type=int, default=None)
    n.add_argument("--force", action="store_true", help="measure even if gate 1 fails (never promotes)")
    n.add_argument("--no-integrity", action="store_true")
    n.set_defaults(fn=cmd_bench)

    g = sub.add_parser("gate", help="editable-surface check only, no measurement")
    g.add_argument("--target", required=True)
    g.set_defaults(fn=cmd_gate)

    p = sub.add_parser("promote", help="referee: append a passing record to the frontier")
    p.add_argument("--target", required=True)
    p.add_argument("--record", required=True, help="path to a results/*.json bench record")
    p.add_argument("--author", required=True, help="GitHub handle of the contributor")
    p.add_argument("--author-name", default=None)
    p.add_argument("--model", default=None, help="the model that wrote it, if any (e.g. 'Claude Opus 5')")
    p.add_argument("--note", default=None, help="one-line summary, or a path to a note file")
    p.add_argument("--url", default=None, help="PR or submission URL")
    p.add_argument("--scope", choices=["engine-general", "model-specific"], default="model-specific")
    p.add_argument("--also-moved", nargs="*", default=None, help="other targets this moved")
    p.add_argument("--referee", default=None)
    p.add_argument("--held-out-record", default=None,
                   help="path to a passing `heldout` verification record (gate 3)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_promote)

    h = sub.add_parser("heldout",
                       help="gate 3: token identity AND speedup on freshly generated prompts")
    h.add_argument("--target", required=True)
    h.add_argument("--count", type=int, default=6, help="how many prompts to generate")
    h.add_argument("--seed", default=None,
                   help="hex seed; omit for a fresh random one. Pass a recorded "
                        "seed to reproduce a past verification.")
    h.add_argument("--claimed-speedup", type=float, default=None,
                   # argparse runs help through %-formatting, so the percent sign
                   # in the tolerance has to be doubled.
                   help="the decodeSpeedup from the bench record you are refereeing. "
                        "The held-out arms must reproduce at least "
                        f"{held_out_tolerance() * 100:.0f}%% of that gain, or gate 3 fails.")
    h.add_argument("--pairs", type=int, default=None,
                   help="paired arms to run (default 1). Raise it when a claim sits "
                        "close to the tolerance.")
    h.add_argument("--referee", default=None)
    h.set_defaults(fn=cmd_heldout)

    l = sub.add_parser("leaderboard", help="print the frontier and its promotion chain")
    l.add_argument("--target", required=True)
    l.add_argument("--init", action="store_true", help="create it from baseline.json if absent")
    l.set_defaults(fn=cmd_leaderboard)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
