#!/bin/zsh
K=~/.local/bin/kaggle
LOG=/Users/kellyhe/Documents/gauntlet/knee/work/v3/poll.log
while true; do
  A=$($K kernels status kellyhe47/knee-cnn-v3a 2>&1 | tail -1)
  B=$($K kernels status kellyhe47/knee-cnn-v3b 2>&1 | tail -1)
  echo "$(date '+%H:%M:%S') A: $A | B: $B" >> $LOG
  S="$A$B"
  if [[ "${S:u}" == *RUNNING* || "${S:u}" == *QUEUED* ]]; then
    sleep 300
  else
    echo "$(date '+%H:%M:%S') TERMINAL" >> $LOG
    break
  fi
done
