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
        "LogMerge_benchmark.py",
        "Drain_benchmark.py",
        "Brain_benchmark.py",
        "LogGzip_benchmark.py",
        "Logram_benchmark.py",
        "MoLFI_benchmark.py",
        "Tipping_benchmark.py",
        "XDrain_benchmark.py",
        "get_metrics.py",
    ]

    for script in benchmark_scripts:
        run_cmd(f"python {script}")
