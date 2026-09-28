#!/usr/bin/env python3
"""
Web viewer for the runs in cases/: images over the steps, outputs, loss, and
arguments, newest first. The page polls the server, so new and running cases
update without reloading.
"""

import argparse
import json
import time
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
# A run without the done record counts as stopped if its log did not change
# within this many seconds, or within three intervals between the checks.
RUNNING_SEC = 90

app = FastAPI()
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class Log:
    """Records of train.jsonl, read incrementally since the file is only appended to during a run."""

    def __init__(self):
        self.first = b""
        self.offset = 0
        self.start = {}
        self.rows = []
        self.done = False

    def update(self, path):
        with open(path, "rb") as f:
            first = f.readline()
            # A new run truncates the file and starts with a different record.
            if first != self.first or f.seek(0, 2) < self.offset:
                self.__init__()
                self.first = first
            f.seek(self.offset)
            data = f.read()
        # An incomplete last line is read at the next update.
        end = data.rfind(b"\n") + 1
        self.offset += end
        for line in data[:end].splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "start":
                self.start = rec
            elif rec.get("event") == "done":
                self.done = True
            elif "step" in rec:
                self.rows.append(rec)
        return self


_logs: dict[Path, Log] = {}


def read_log(case):
    path = case / "train.jsonl"
    if not path.exists():
        _logs.pop(path, None)
        return Log()
    return _logs.setdefault(path, Log()).update(path)


def read_config(case):
    path = case / "config.yaml"
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return {}


def case_dir(name):
    case = (CASES / name).resolve()
    if case.parent != CASES or not case.is_dir():
        raise HTTPException(404, f"no case {name!r}")
    return case


def summary(case, log):
    config = read_config(case)
    rows = log.rows
    target = config.get("target")
    hits = [r["step"] for r in rows if target is not None and r.get("output") == target]
    path = case / "train.jsonl"
    # Time of the last change of the log, of the edit of run.sh if not run yet.
    mtime = next(f.stat().st_mtime for f in [path, case / "run.sh", case] if f.exists())
    if log.done:
        status = "done"
    elif path.exists():
        times = [r["time"] for r in rows[-2:]]
        interval = times[-1] - times[0] if len(times) == 2 else 0
        status = "running" if time.time() - mtime < max(RUNNING_SEC, 3 * interval) else "stopped"
    else:
        status = None
    images = [r["step"] for r in rows if r.get("image") and (case / r["image"]).exists()]
    last = rows[-1] if rows else {}
    return {
        "name": case.name,
        "mtime": mtime,
        "status": status,
        "running": status == "running",
        "target": target,
        "method": config.get("method"),
        "model": config.get("model"),
        "steps": config.get("steps"),
        "step": last.get("step"),
        "loss": last.get("loss"),
        "tokens_ok": last.get("tokens_ok"),
        "ms_step": last.get("ms_step"),
        "output": last.get("output"),
        "checks": len(rows),
        "hits": len(hits),
        "first_hit": hits[0] if hits else None,
        "last_hit": bool(rows) and rows[-1].get("output") == target,
        "image": images[-1] if images else None,
        "images": images,
        # Id of the run, for caching of images.
        "version": log.start.get("run", "0"),
    }


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/runs")
def runs():
    cases = [c for c in CASES.iterdir() if c.is_dir()] if CASES.exists() else []
    summaries = []
    for case in cases:
        s = summary(case, read_log(case))
        del s["images"]
        summaries.append(s)
    return sorted(summaries, key=lambda s: s["mtime"], reverse=True)


@app.get("/api/runs/{name}")
def run(name: str):
    case = case_dir(name)
    run_sh = case / "run.sh"
    log = read_log(case)
    return {
        **summary(case, log),
        "config": read_config(case),
        "run_sh": run_sh.read_text() if run_sh.exists() else None,
        "log": log.rows,
    }


@app.get("/img/{name}/{step}.png")
def image(name: str, step: int):
    path = case_dir(name) / f"metamer_{step:05d}.png"
    if not path.exists():
        raise HTTPException(404)
    # Images of a step do not change within a run, and the URL has the id of the run.
    return FileResponse(path, headers={"Cache-Control": "max-age=86400"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    print(f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
