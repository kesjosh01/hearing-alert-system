#!/usr/bin/env python3
"""Summarize hardware log evidence after confirmed C3 Light-sleep entry."""
import argparse
import json
import re
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("log", type=Path)
parser.add_argument("--events", type=int, default=100)
args = parser.parse_args()
if args.events < 1:
    parser.error("--events must be positive")
lines = re.sub(r"\x1b\[[0-9;]*m", "", args.log.read_text(errors="replace")).splitlines()
samples = []
confirmed = False
first_sleep = None
last_sleep = None
transmissions = {}
acks = {}
failed = set()
for line in lines:
    pm = re.search(r"C3_PM session=(\d+) sleep_count=(\d+) reject_count=(\d+)", line)
    if pm:
        item = dict(zip(("session", "successful", "rejected"), map(int, pm.groups())))
        samples.append(item)
        if item["successful"] > 0:
            confirmed = True
            if first_sleep is None:
                first_sleep = item
        last_sleep = item
    if "receiver registered" in line or "heartbeat expired" in line:
        # A new registration needs its own sleep evidence, even before the first TX.
        if transmissions:
            raise SystemExit("Registration changed within sample; split logs by test run first")
        confirmed = False
        first_sleep = last_sleep = None
        samples.clear()
    if not confirmed:
        continue
    tx = re.search(r"TX seq=(\d+) attempt=(\d+)", line)
    if tx:
        seq, attempt = map(int, tx.groups())
        if seq in transmissions or len(transmissions) < args.events:
            if seq not in transmissions and attempt != 1:
                raise SystemExit("Sample starts with a retry; capture the first TX attempt")
            transmissions.setdefault(seq, []).append(attempt)
    ack = re.search(r"ACK seq=(\d+) elapsed=(\d+)ms attempts=(\d+)", line)
    if ack and int(ack[1]) in transmissions:
        acks[int(ack[1])] = {"elapsed_ms": int(ack[2]), "attempts": int(ack[3])}
    failure = re.search(r"FAILED seq=(\d+)", line)
    if failure and int(failure[1]) in transmissions:
        failed.add(int(failure[1]))
sessions = {item["session"] for item in samples}
times = [item["elapsed_ms"] for item in acks.values()]
result = {
    "log": str(args.log),
    "requested_events": args.events,
    "transmitted_unique": len(transmissions),
    "acknowledged_unique": len(acks),
    "sequence_first": min(transmissions) if transmissions else None,
    "sequence_last": max(transmissions) if transmissions else None,
    "missing_ack_sequences": sorted(set(transmissions) - set(acks)),
    "retransmissions": sum(len(attempts) - 1 for attempts in transmissions.values()),
    "failed_events": sorted(failed),
    "ack_round_trip_ms": {
        "min": min(times) if times else None,
        "mean": round(mean(times), 2) if times else None,
        "max": max(times) if times else None,
    },
    "first_confirmed_sleep": first_sleep,
    "last_sleep": last_sleep,
    "single_c3_session": len(sessions) == 1,
    "sleep_counter_increased": bool(first_sleep and last_sleep
        and first_sleep["session"] == last_sleep["session"]
        and last_sleep["successful"] > first_sleep["successful"]),
}
result["passed"] = (len(transmissions) == args.events and len(acks) == args.events
    and not failed and len(sessions) == 1 and result["sleep_counter_increased"])
print(json.dumps(result, indent=2, ensure_ascii=False))
raise SystemExit(0 if result["passed"] else 1)
