#!/bin/bash
set -e                  # ensures cron gets a non-zero exit code if sbatch fails.
cd /home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/submission
sbatch run_pancast.sh