#!/usr/bin/env python3
import os

def read_log_tail(log_path, lines=50):
    """Read the last N lines of a log file"""
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    print(f"=== Last {lines} lines of {log_path} ===")
    for line in all_lines[-lines:]:
        print(line.strip())

if __name__ == "__main__":
    log_file = "logs/multi_speaker_results_20251222_194144.txt"
    read_log_tail(log_file)