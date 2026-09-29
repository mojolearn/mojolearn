#!/bin/bash
# The two gates this lane's edit of the fragment loads must pass before any
# time is read (DEVIATION 2975): the fifteen-bit gate and the existing
# low-bit gate, each with its arms, on the shared NVIDIA pod.
exec bash "$(dirname "$0")/box_job.sh" h100 gate int8_gate sim
