#!/usr/bin/env bash
# Network-only chaining (no computation on this laptop):
# wait for the dataset on Kaggle -> start the final job -> wait -> download results.
cd /c/Padel
export PYTHONIOENCODING=utf-8
K="python -m kaggle"
OUT=kaggle_outputs
mkdir -p $OUT
log() { echo "$(date '+%H:%M:%S') $*" | tee -a $OUT/finish.log; }

until $K datasets files hashulearn/padel-coach-data 2>&1 | grep -q "2022_BCN_FinalM_1.mp4"; do sleep 120; done
until $K datasets files hashulearn/padel-coach-models 2>&1 | grep -q "pose_best.pt"; do sleep 60; done
log "datasets ready"
sleep 120   # let Kaggle finish processing the files

until r=$($K kernels push -p kaggle/infer --accelerator NvidiaTeslaT4 2>&1); echo "$r" | grep -q "successfully pushed"; do
  log "push refused: $r"
  echo "$r" | grep -qiE "quota|verif|phone" && { log "STOP: account cannot use GPU/internet"; exit 1; }
  sleep 300
done
log "final job started"

while true; do
  s=$($K kernels status hashulearn/padel-coach-inference 2>&1 | tail -1)
  case "$s" in
    *COMPLETE*) log "final job complete"; break ;;
    *ERROR*|*CANCEL*) log "final job FAILED: $s"; $K kernels logs hashulearn/padel-coach-inference > $OUT/padel-coach-inference.log 2>&1; exit 1 ;;
    *) sleep 300 ;;
  esac
done
mkdir -p $OUT/padel-coach-inference
$K kernels output hashulearn/padel-coach-inference -p $OUT/padel-coach-inference > /dev/null 2>&1
$K kernels logs hashulearn/padel-coach-inference > $OUT/padel-coach-inference.log 2>&1
log "ALL DONE - results in $OUT/padel-coach-inference"
