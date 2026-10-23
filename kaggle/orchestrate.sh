#!/usr/bin/env bash
# Drives the Kaggle training end to end:
#  1. keeps retrying the pose/shot kernel until a GPU slot is free
#  2. waits for both training kernels, downloads their outputs
#  3. runs the end-to-end inference kernel on both finals, downloads detections
# Progress is appended to kaggle_outputs/orchestrate.log
cd /c/Padel
K="python -m kaggle"
OUT=kaggle_outputs
mkdir -p $OUT
log() { echo "$(date '+%H:%M:%S') $*" | tee -a $OUT/orchestrate.log; }
status() { $K kernels status "hashuu7/$1" 2>&1 | tail -1; }

# 1. start pose kernel when allowed
if ! status padel-coach-train-pose-shot | grep -qE "RUNNING|COMPLETE|QUEUED"; then
  until $K kernels push -p kaggle/pose --accelerator NvidiaTeslaT4 2>&1 | tee -a $OUT/orchestrate.log | grep -q "successfully pushed"; do
    log "pose kernel: GPU slot busy, retrying in 2 min"; sleep 120
  done
  log "pose kernel started"
fi

# 2. wait for both training kernels
for k in padel-coach-train-ball padel-coach-train-pose-shot; do
  while true; do
    s=$(status $k)
    case "$s" in
      *COMPLETE*) log "$k complete"; break ;;
      *ERROR*|*CANCEL*) log "$k FAILED: $s"; $K kernels logs hashuu7/$k > $OUT/$k.log 2>&1; log "log saved to $OUT/$k.log"; exit 1 ;;
      *) sleep 300 ;;
    esac
  done
  mkdir -p $OUT/$k
  $K kernels output hashuu7/$k -p $OUT/$k >> $OUT/orchestrate.log 2>&1
  $K kernels logs hashuu7/$k > $OUT/$k.log 2>&1
  log "$k outputs downloaded"
done

# 3. end-to-end inference on both finals with the trained models
until $K kernels push -p kaggle/infer --accelerator NvidiaTeslaT4 2>&1 | tee -a $OUT/orchestrate.log | grep -q "successfully pushed"; do
  log "inference kernel: waiting for GPU slot"; sleep 300
done
log "inference kernel started"
while true; do
  s=$(status padel-coach-inference)
  case "$s" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCEL*) log "inference FAILED: $s"; $K kernels logs hashuu7/padel-coach-inference > $OUT/padel-coach-inference.log 2>&1; exit 1 ;;
    *) sleep 300 ;;
  esac
done
mkdir -p $OUT/padel-coach-inference
$K kernels output hashuu7/padel-coach-inference -p $OUT/padel-coach-inference >> $OUT/orchestrate.log 2>&1
$K kernels logs hashuu7/padel-coach-inference > $OUT/padel-coach-inference.log 2>&1
log "ALL DONE"
