#!/usr/bin/env python3
"""
spawn_my_script.py

Launches 10 parallel Python processes running `my_file.py`.
"""

import subprocess
import sys

def main():
    # name of the script you want to run in parallel
    target_script = r"C:\Users\kylan\Documents\code\repos\KOTron\tron-python\scripts\y2025\m08\client.py"
    # how many processes to spawn
    num_procs = 7

    procs = []
    for i in range(num_procs):
        # Use the same Python interpreter that’s running this wrapper
        p = subprocess.Popen([sys.executable, target_script])
        procs.append(p)
        print(f"Started process #{i+1} (PID={p.pid})")

    # Wait for all to finish
    for p in procs:
        ret = p.wait()
        print(f"Process PID={p.pid} exited with code {ret}")

if __name__ == '__main__':
    main()
