#!/bin/zsh
K=~/.local/bin/kaggle
LOG=/Users/kellyhe/Documents/gauntlet/knee/work/v3/poll_submit.log
while true; do
  S=$($K kernels status kellyhe47/knee-submit-v1 2>&1 | tail -1)
  echo "$(date '+%H:%M:%S') $S" >> $LOG
  if [[ "${S:u}" == *RUNNING* || "${S:u}" == *QUEUED* ]]; then
    sleep 180
  else
    echo "$(date '+%H:%M:%S') TERMINAL" >> $LOG
    break
  fi
done
