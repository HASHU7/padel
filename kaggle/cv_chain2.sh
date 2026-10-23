#!/usr/bin/env bash
# Network-only: finish the cross-match evaluation (no computation on this laptop).
cd /c/Padel
export PYTHONIOENCODING=utf-8
K="python -m kaggle"; O=hashulearn; OUT=kaggle_outputs/cv
log() { echo "$(date '+%H:%M:%S') $*" | tee -a $OUT/cv_chain.log; }
st() { $K kernels status "$O/$1" 2>&1 | tail -1; }
wait_done() {
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
wait_done padel-cv-ball-women && fetch padel-cv-ball-women
wait_done padel-cv-ball-men && fetch padel-cv-ball-men
until r=$($K kernels push -p kaggle/cv_e2e --accelerator NvidiaTeslaT4 2>&1); echo "$r" | grep -q "successfully pushed"; do
  echo "$r" | grep -qi quota && { log "STOP: GPU quota reached"; exit 1; }
  sleep 180
done
log "cv_e2e started"
wait_done padel-cv-e2e && fetch padel-cv-e2e
log "ALL CV DONE"
