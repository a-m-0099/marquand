#!/bin/sh
# memguard.sh CMD...: run CMD, kill it if MemAvailable drops under 2.5 GB (protects the desktop from OOM).
"$@" & P=$!
while kill -0 $P 2>/dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$a" -lt 2500 ]; then echo "memguard: MemAvailable ${a}MB, killing $P" >&2; kill -9 $P; fi
  sleep 0.5
done
wait $P
