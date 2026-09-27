# pauses long jobs (training, benchmarks) while the laptop is hot, reading the CPU and GPU sensors
import glob, os, sys, time

LIMIT = float(os.environ.get("MARQ_MAX_TEMP", "85"))
RESUME = float(os.environ.get("MARQ_RESUME_TEMP", str(LIMIT - 10)))


def hottest():
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
            except (OSError, ValueError):  # a sleeping GPU has no readings
                pass
    return max(temps) if temps else None


def guard(limit=LIMIT, resume=RESUME, read=hottest, sleep=time.sleep):
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
