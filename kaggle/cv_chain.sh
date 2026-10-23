#!/usr/bin/env bash
# Network-only chaining for the cross-match evaluation (no computation on this laptop).
cd /c/Padel
export PYTHONIOENCODING=utf-8
K="python -m kaggle"
O=hashulearn
OUT=kaggle_outputs/cv
mkdir -p $OUT
log() { echo "$(date '+%H:%M:%S') $*" | tee -a $OUT/cv_chain.log; }
st() { $K kernels status "$O/$1" 2>&1 | tail -1; }
wait_done() {  # $1 = kernel slug
  while true; do
    s=$(st $1)
    case "$s" in
      *COMPLETE*) log "$1 complete"; return 0 ;;
      *ERROR*|*CANCEL*) log "$1 FAILED: $s"; $K kernels logs $O/$1 > $OUT/$1.log 2>&1; return 1 ;;
      *) sleep 300 ;;
    esac
  done
}
fetch() { mkdir -p $OUT/$1; $K kernels output $O/$1 -p $OUT/$1 > /dev/null 2>&1; $K kernels logs $O/$1 > $OUT/$1.log 2>&1; log "$1 downloaded"; }
push_gpu() {  # retry until a GPU slot is free
  until r=$($K kernels push -p kaggle/$1 --accelerator NvidiaTeslaT4 2>&1); echo "$r" | grep -q "successfully pushed"; do
    echo "$r" | grep -qi quota && { log "STOP: GPU quota reached ($1)"; exit 1; }
    sleep 180
  done
  log "$1 started"
}

wait_done padel-cv-shot && fetch padel-cv-shot
wait_done padel-cv-pose-women; fetch padel-cv-pose-women
push_gpu cv_ball_women
wait_done padel-cv-pose-men; fetch padel-cv-pose-men
push_gpu cv_ball_men
wait_done padel-cv-ball-women; fetch padel-cv-ball-women
wait_done padel-cv-ball-men; fetch padel-cv-ball-men
log "ALL CV DONE"
