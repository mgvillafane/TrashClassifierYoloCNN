"""Mirror Databricks epoch logs into a live local TensorBoard browser tab."""

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

from sync_taco_databricks import find_cli
from taco_yolo import ROOT, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--remote-run", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--profile", default="Marcos Gomez Villafane")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args(argv)
    local = args.local_dir.resolve()
    local.mkdir(parents=True, exist_ok=True)
    if args.background:
        forwarded = [value for value in (sys.argv[1:] if argv is None else argv) if value != "--background"]
        with (local / "watcher.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", "-B", str(Path(__file__).resolve()), *forwarded],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        print(json.dumps({"watcher_pid": process.pid, "log": str(local / "watcher.log")}))
        return
    cli = find_cli()
    remote = args.remote_run.removeprefix("dbfs:").rstrip("/")
    if not remote.startswith("/Volumes/"):
        raise ValueError("--remote-run must point to a Unity Catalog volume directory")

    def command(arguments):
        result = subprocess.run([cli, *arguments, "-p", args.profile], cwd=ROOT,
                                capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError(result.stderr.strip()[-1000:])
        return result.stdout

    port = args.port
    while port < args.port + 20:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                break
            except OSError:
                port += 1
    else:
        raise RuntimeError("No free TensorBoard port found")
    with (local / "tensorboard.log").open("a", encoding="utf-8") as log:
        tensorboard = subprocess.Popen([
            sys.executable, "-B", "-m", "tensorboard.main", "--logdir", str(local),
            "--host", "127.0.0.1", "--port", str(port), "--load_fast=false", "--reload_interval", "5",
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    url = f"http://localhost:{port}/#scalars"
    metadata = {"run_id": args.run_id, "remote_run": remote, "tensorboard_url": url,
                "tensorboard_pid": tensorboard.pid, "watcher_pid": __import__("os").getpid(),
                "profile": args.profile, "local_dir": str(local)}
    write_json(local / "live_viewer.json", metadata)
    for _ in range(60):
        if tensorboard.poll() is not None:
            raise RuntimeError(f"TensorBoard exited; inspect {local / 'tensorboard.log'}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                if response.status == 200:
                    break
        except (OSError, TimeoutError):
            time.sleep(1)
    else:
        raise RuntimeError("TensorBoard did not start within 60 seconds")
    if args.open_browser:
        metadata["browser_opened"] = webbrowser.open_new_tab(url)
        write_json(local / "live_viewer.json", metadata)
    print(f"TensorBoard ready: {url}", flush=True)
    changed = {}
    last_state = None
    mirrored_events = 0

    def mirror():
        nonlocal mirrored_events
        entries = json.loads(command(["fs", "ls", f"dbfs:{remote}", "-o", "json"]))
        for entry in entries:
            name = entry["name"]
            if entry.get("is_directory") or not (name.startswith("events.out.tfevents.")
                    or name in {"results.csv", "training_status.json", "artifacts.json", "run_config.json"}):
                continue
            target = local / name
            # Events are immutable and only uploaded after their writer closes.
            if name.startswith("events.out.tfevents.") and target.is_file():
                continue
            fingerprint = (entry.get("size"), entry.get("last_modified"))
            if changed.get(name) == fingerprint and target.is_file():
                continue
            command(["fs", "cp", f"dbfs:{remote}/{name}", str(target), "--overwrite"])
            changed[name] = fingerprint
            if name.startswith("events.out.tfevents."):
                mirrored_events += 1
                print(f"Mirrored TensorBoard event: {name}", flush=True)

    while True:
        finished = False
        try:
            state = json.loads(command(["jobs", "get-run", args.run_id, "-o", "json"]))
            compact = {"state": state.get("state"), "run_page_url": state.get("run_page_url"),
                       "run_id": state.get("run_id"), "tasks": [
                           {"task_key": task.get("task_key"), "run_id": task.get("run_id"),
                            "state": task.get("state")} for task in state.get("tasks", [])]}
            write_json(local / "databricks_run.json", compact)
            current = json.dumps(compact["state"], sort_keys=True)
            if current != last_state:
                print(f"Databricks state: {current}", flush=True)
                last_state = current
            finished = (state.get("state") or {}).get("life_cycle_state") in {
                "TERMINATED", "SKIPPED", "INTERNAL_ERROR"}
        except Exception as exc:
            print(f"Waiting for Databricks status: {exc}", flush=True)
        try:
            mirror()
        except Exception as exc:
            print(f"Waiting for epoch logs: {exc}", flush=True)
        write_json(local / "mirror_status.json", {
            "mirrored_event_files": mirrored_events, "last_poll_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "training_run_finished": finished, "tensorboard_url": url})
        if finished:
            print("Databricks run finished. TensorBoard remains available in the browser.", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
