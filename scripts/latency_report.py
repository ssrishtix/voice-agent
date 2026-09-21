"""Summarize data/latency.csv. From the project root:

    .venv\\Scripts\\python.exe scripts\\latency_report.py
"""
import csv
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
path = os.path.join("data", "latency.csv")
if not os.path.exists(path):
    sys.exit("no data/latency.csv yet — make a call first")
vals = sorted(int(r[2]) for r in csv.reader(open(path)) if r)
if not vals:
    sys.exit("data/latency.csv is empty")
p95 = vals[min(len(vals) - 1, int(len(vals) * 0.95))]
print(f"turns: {len(vals)}  min: {vals[0]} ms  median: {int(statistics.median(vals))} ms  p95: {p95} ms  max: {vals[-1]} ms")
