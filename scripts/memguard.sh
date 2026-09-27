#!/bin/sh
# memguard.sh CMD... - runs CMD and kills it if free RAM drops under 2.5 GB, so it can't take the desktop down
"$@" & P=$!
while kill -0 $P 2>/dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$a" -lt 2500 ]; then echo "memguard: MemAvailable ${a}MB, killing $P" >&2; kill -9 $P; fi
  sleep 0.5
done
wait $P
