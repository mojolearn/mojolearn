#!/bin/bash
cd "$(dirname "$0")/.."
exec bash tools/pr44_gpu.sh nvidia
