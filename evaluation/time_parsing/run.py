import subprocess
import sys

def run_cmd(cmd):
    print(f"[RUNNING] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    benchmark_scripts = [
        # "LogMerge_timeparsing.py",
        # "Drain_timeparsing.py",
        # "Brain_timeparsing.py",
        # "LogGzip_timeparsing.py",
        "Logram_timeparsing.py",
        "MoLFI_timeparsing.py",
        "Tipping_timeparsing.py",
        "XDrain_timeparsing.py",
    ]

    for script in benchmark_scripts:
        run_cmd(f"python {script}")
