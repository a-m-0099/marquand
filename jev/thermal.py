"""Thermal guard for long GPU jobs (training, benchmarks): pause while the laptop is hot.

Reads the CPU (k10temp Tctl) and every amdgpu sensor (edge, junction, memory) from hwmon; a suspended dGPU simply
has no readings. Limits come from JEV_MAX_TEMP / JEV_RESUME_TEMP (default 85 / 75 C).
"""
import glob, os, sys, time

LIMIT = float(os.environ.get("JEV_MAX_TEMP", "85"))
RESUME = float(os.environ.get("JEV_RESUME_TEMP", str(LIMIT - 10)))


def hottest():
    """Hottest CPU/GPU temperature in C, or None when no sensor is readable."""
    temps = []
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(f"{h}/name").read().strip() not in ("k10temp", "amdgpu"):
                continue
        except OSError:
            continue
        for f in glob.glob(f"{h}/temp*_input"):
            try:
                temps.append(int(open(f).read()) / 1000)
            except (OSError, ValueError):  # sensor of a device in D3cold
                pass
    return max(temps) if temps else None


def guard(limit=LIMIT, resume=RESUME, read=hottest, sleep=time.sleep):
    """Block while hot: once `limit` is reached, wait until it drops to `resume`. Returns whether it waited."""
    t = read()
    if t is None or t < limit:
        return False
    print(f"thermal: {t:.0f}C >= {limit:.0f}C, pausing until {resume:.0f}C", file=sys.stderr, flush=True)
    while True:
        sleep(5)
        t = read()
        if t is None or t <= resume:
            print(f"thermal: {t}C, resuming", file=sys.stderr, flush=True)
            return True
