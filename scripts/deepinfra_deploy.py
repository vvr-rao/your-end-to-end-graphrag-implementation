#!/usr/bin/env python3
"""Create / inspect / delete the DEDICATED Qwen3.8-Flash-Next deployment.

    uv run python scripts/deepinfra_deploy.py create --from-sizing .build/sizing.json [--name flash-next]
    uv run python scripts/deepinfra_deploy.py status
    uv run python scripts/deepinfra_deploy.py delete <deploy_id>

PAID: a deployment bills per GPU-minute from creation until deletion,
whether or not anything calls it. `create` prints the configuration and
hourly rate and asks for a typed "yes" (or --yes, which the skills pass
only after the user approved in chat). Every create/delete is appended to
the GPU ledger (--ledger, default .build/gpu_ledger.jsonl) so actual
GPU-hours can be reconciled against the sizing estimate.

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
        raise SystemExit(
            f"{method} {path} -> HTTP {exc.code}: {body}"
        ) from None
    return json.loads(raw) if raw else {}


def build_payload(sizing: dict, name: str, max_batch_size: int | None = None) -> dict:
    """The POST /deploy/llm body for the sized option. Pure (unit-tested)."""
    opt = sizing["option"]
    conc = sizing.get("concurrency") or {}
    batch = max_batch_size or max(8, min(256, max(conc.values()) if conc else 64))
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
        sizing = json.loads(Path(_arg(argv, "--from-sizing", str(ROOT / ".build" / "sizing.json"))).read_text())
        mbs = _arg(argv, "--max-batch-size")
        payload = build_payload(sizing, _arg(argv, "--name", "flash-next"), int(mbs) if mbs else None)
        opt = sizing["option"]
        print(json.dumps(payload, indent=2))
        print(f"\nThis bills ${opt['usd_hr']:.2f}/hour ({opt['label']}) until deleted. "
              f"Sized estimate for the corpus: {sizing['estimate']['hours']} h, ~${sizing['estimate']['usd']}.")
        if "--yes" not in argv and input('Type "yes" to create: ').strip().lower() != "yes":
            print("not created.")
            return 1
        res = _call("POST", "/deploy/llm", payload)
        did = res.get("deploy_id") or res.get("id")
        ledger(led, {"event": "create", "deploy_id": did, "usd_hr": opt["usd_hr"],
                     "label": opt["label"], "payload": payload})
        print(json.dumps(res, indent=2))
        print(f"\nModel id: <your DeepInfra username>/{payload['model_name']}  (or deploy_id:{did})")
        print("Set DEEPINFRA_FLASH_MODEL in .env, then: uv run python scripts/select_llm_mode.py deepinfra")
        print("Wait for `status` to show it running, then: deepinfra_sizing.py --probe")
        return 0
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
