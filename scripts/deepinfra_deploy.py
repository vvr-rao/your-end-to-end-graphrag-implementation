#!/usr/bin/env python3
"""Create / inspect / delete the DEDICATED Qwen3.8-27B deployment.

    uv run python scripts/deepinfra_deploy.py create --from-sizing .build/sizing.json \
        [--name qwen27b] [--apply-concurrency] [--no-fallback] [--wait-min 30]
    uv run python scripts/deepinfra_deploy.py status
    uv run python scripts/deepinfra_deploy.py delete <deploy_id>

PAID: a deployment bills per GPU-minute from creation until deletion,
whether or not anything calls it. `create` prints every configuration it may
try with its hourly rate and asks for a typed "yes" (or --yes, which the
skills pass only after the user approved in chat). Every create/delete is
appended to the GPU ledger (--ledger, default .build/gpu_ledger.jsonl) so
actual GPU-hours can be reconciled against the sizing estimate.

CAPACITY: DeepInfra often has no free GPUs of a type -- the create answers
409 "currently unavailable", or the deployment is accepted and then fails
with `no-gpu-available`. `create` walks the sizing file's preference chain
(2x B200 -> 2x H200 -> 2x H100), waits for each deployment to reach
`running`, deletes any that fail, and stops at the first that runs. It
records which one in the sizing file (`active`), and --apply-concurrency
writes THAT configuration's concurrency into config/config.yaml. Exit 3 =
no configuration had capacity (nothing left running); retry later.

After `create`, put the printed model id in .env as DEEPINFRA_FLASH_MODEL
(an identifier, not a secret) and re-run select_llm_mode.py deepinfra.

Reads DEEPINFRA_API_KEY from settings; never prints it.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.deepinfra.com"
DEFAULT_LEDGER = ROOT / ".build" / "gpu_ledger.jsonl"


def _key() -> str:
    from backend.app.core.config import get_settings

    key = get_settings().deepinfra_api_key
    if not key:
        raise SystemExit("DEEPINFRA_API_KEY is not set in .env (add it yourself; never paste it in chat)")
    return key


class DeployHTTPError(SystemExit):
    """A non-2xx answer from the deploy API (still exits the CLI by default)."""

    def __init__(self, method: str, path: str, code: int, body: str) -> None:
        super().__init__(f"{method} {path} -> HTTP {code}: {body}")
        self.code, self.body = code, body


def _no_capacity(exc: DeployHTTPError) -> bool:
    return exc.code == 409 and "unavailable" in exc.body.lower()


def _call(method: str, path: str, body: dict | None = None) -> dict | list:
    req = urllib.request.Request(
        API + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()[:400].decode(errors="replace")
        if exc.code == 409 and "display name" in body:
            # A dedicated model is addressed as "<username>/<model_name>", so
            # the account needs a username before anything can be deployed.
            raise SystemExit(
                "DeepInfra refused: the account has no display name (username). Set one "
                "in your DeepInfra account/profile settings, then re-run. Nothing was created."
            ) from None
        raise DeployHTTPError(method, path, exc.code, body) from None
    return json.loads(raw) if raw else {}


def build_payload(sizing: dict, name: str, max_batch_size: int | None = None) -> dict:
    """The POST /deploy/llm body for one sized option (the sizing file's top
    level, or one of its `fallbacks`). Pure (unit-tested)."""
    opt = sizing["option"]
    conc = sizing.get("concurrency") or {}
    batch = max_batch_size or sizing.get("max_batch_size") or max(
        8, min(256, max(conc.values()) if conc else 64))
    return {
        "model_name": name,
        "gpu": opt["gpu"],
        "num_gpus": int(opt["num_gpus"]),
        "max_batch_size": int(batch),
        "hf": {"repo": opt["repo"]},
        # min == max: a benchmark/ingest window wants the capacity it sized
        # for, not scale-to-zero cold starts mid-stage. Delete when done.
        "settings": {"min_instances": int(opt["instances"]), "max_instances": int(opt["instances"])},
    }


def candidates(sizing: dict, fallback: bool = True) -> list[dict]:
    """The configurations `create` may try, in preference order: the sizing
    file's recommendation, then its `fallbacks` (each carries its own
    option, estimate, concurrency and max_batch_size)."""
    head = {k: sizing[k] for k in ("option", "estimate", "concurrency", "max_batch_size") if k in sizing}
    return [head] + (list(sizing.get("fallbacks") or []) if fallback else [])


def deploy_name(base: str, opt: dict, stamp: str) -> str:
    """Deployment names stay reserved after deletion, so every attempt gets a
    fresh one: <base>-<gpu>-<stamp>, e.g. qwen27b-b200-10171830."""
    return f"{base}-{opt['gpu'].split('-')[0].lower()}-{stamp}"


def deploy_state(did: str) -> tuple[str, str]:
    rows = _call("GET", "/deploy/list")
    rows = rows if isinstance(rows, list) else rows.get("deployments", rows)
    r = next((r for r in rows if r.get("deploy_id") == did), None) or {}
    return r.get("status", "missing"), r.get("fail_reason") or ""


def wait_until_running(did: str, minutes: float, poll_s: float = 30.0,
                       state=deploy_state, sleep=time.sleep) -> tuple[str, str]:
    """Poll until `running` or a terminal failure; ("timeout", "") if neither
    within `minutes`. Loading 31 GB of FP8 weights took 4-8 min per run."""
    deadline = time.monotonic() + minutes * 60
    while True:
        st, why = state(did)
        if st in ("running", "failed", "deleted", "missing"):
            return st, why
        if time.monotonic() >= deadline:
            return "timeout", ""
        sleep(poll_s)


def ledger(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **event}) + "\n")


def gpu_hours(path: Path) -> list[dict]:
    """Pair create/delete events per deploy_id -> billed hours and USD."""
    if not path.exists():
        return []
    opened: dict[str, dict] = {}
    out: list[dict] = []
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "create":
            opened[e["deploy_id"]] = e
        elif e["event"] == "delete" and e["deploy_id"] in opened:
            c = opened.pop(e["deploy_id"])
            t0 = time.mktime(time.strptime(c["ts"][:19], "%Y-%m-%dT%H:%M:%S"))
            t1 = time.mktime(time.strptime(e["ts"][:19], "%Y-%m-%dT%H:%M:%S"))
            h = (t1 - t0) / 3600
            out.append({"deploy_id": e["deploy_id"], "hours": round(h, 3),
                        "usd": round(h * c["usd_hr"], 2), "label": c.get("label")})
    for did, c in opened.items():
        out.append({"deploy_id": did, "hours": None, "usd": None, "label": c.get("label"),
                    "note": "STILL RUNNING (billing)"})
    return out


def _arg(argv: list[str], name: str, default=None):
    if name in argv and argv.index(name) + 1 < len(argv):
        return argv[argv.index(name) + 1]
    return default


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd = argv[0]
    led = Path(_arg(argv, "--ledger", str(DEFAULT_LEDGER)))

    if cmd == "create":
        sizing_path = Path(_arg(argv, "--from-sizing", str(ROOT / ".build" / "sizing.json")))
        sizing = json.loads(sizing_path.read_text())
        mbs = _arg(argv, "--max-batch-size")
        base = _arg(argv, "--name", "qwen27b")
        cands = candidates(sizing, fallback="--no-fallback" not in argv)
        print("Will try, in order, until one has capacity and reaches `running`:")
        for i, c in enumerate(cands, 1):
            o = c["option"]
            print(f"  {i}. {o['label']}  ${o['usd_hr']:.2f}/hour  (est. {c.get('estimate', {}).get('hours')} h, "
                  f"~${c.get('estimate', {}).get('usd')})")
        print("Each bills from creation until deleted; failed attempts are deleted at once.")
        if "--yes" not in argv and input('Type "yes" to create: ').strip().lower() != "yes":
            print("not created.")
            return 1
        stamp = time.strftime("%m%d%H%M")
        for c in cands:
            opt = c["option"]
            payload = build_payload(c, deploy_name(base, opt, stamp), int(mbs) if mbs else None)
            try:
                res = _call("POST", "/deploy/llm", payload)
            except DeployHTTPError as exc:
                if _no_capacity(exc):
                    print(f"- {opt['label']}: no capacity (409); next")
                    continue
                raise
            did = res.get("deploy_id") or res.get("id")
            ledger(led, {"event": "create", "deploy_id": did, "usd_hr": opt["usd_hr"],
                         "label": opt["label"], "payload": payload})
            print(f"- {opt['label']}: created {did} as {payload['model_name']}; waiting for it to run...")
            if "--no-wait" in argv:
                st, why = "created", ""
            else:
                st, why = wait_until_running(did, float(_arg(argv, "--wait-min", 30)))
            if st not in ("running", "created"):
                print(f"  {st} {why}".rstrip() + "; deleting it")
                _call("DELETE", f"/deploy/{did}")
                ledger(led, {"event": "delete", "deploy_id": did})
                continue
            sizing["active"] = {**c, "deploy_id": did, "model_name": payload["model_name"],
                                "state": st, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            sizing_path.write_text(json.dumps(sizing, indent=2))
            print(f"\n{opt['label']} is {st}: deploy_id {did}, billing ${opt['usd_hr']:.2f}/hour.")
            print(f"Model id: <your DeepInfra username>/{payload['model_name']}  (or deploy_id:{did})")
            if "--apply-concurrency" in argv:
                from tpm_check import apply_concurrency

                path = ROOT / "config" / "config.yaml"
                changes = apply_concurrency(path, c["concurrency"])
                print(f"Concurrency for {opt['label']} -> {path.relative_to(ROOT)}:")
                for line in changes or ["(already at these values)"]:
                    print(f"  {line}")
            else:
                print("Concurrency for this GPU: " + ", ".join(f"{k} {v}" for k, v in sorted(c["concurrency"].items()))
                      + "  (re-run with --apply-concurrency, or apply by hand)")
            print("Set DEEPINFRA_FLASH_MODEL in .env, then: uv run python scripts/select_llm_mode.py deepinfra")
            print("Optional, ~5 GPU-minutes: deepinfra_sizing.py --probe measures the real throughput knee.")
            return 0
        print("\nNo configuration in the chain had capacity; nothing is left running. Retry later.")
        return 3
    if cmd == "status":
        print(json.dumps(_call("GET", "/deploy/list"), indent=2)[:4000])
        for row in gpu_hours(led):
            print(row)
        return 0
    if cmd == "delete":
        if len(argv) < 2:
            print("usage: deepinfra_deploy.py delete <deploy_id>")
            return 2
        did = argv[1]
        res = _call("DELETE", f"/deploy/{did}")
        ledger(led, {"event": "delete", "deploy_id": did})
        print(json.dumps(res, indent=2))
        for row in gpu_hours(led):
            if row["deploy_id"] == did:
                print(f"billed ~{row['hours']} h, ~${row['usd']}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
