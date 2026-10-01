#!/bin/bash
# NVIDIA neural pass then PR #56, sequential on one pod GPU.
bash /root/mojolearn-np47/tools/neural_pass_job.sh nvidia
bash /root/mojolearn-np48/tools/pr56_job.sh nvidia
