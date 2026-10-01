#!/bin/bash
bash /root/mojolearn-pass30/tools/cell_ab_job2.sh nvidia pagerank "x_neighbors" "pagerank" "" MOJOLEARN_PR_SPARSE=0 "istella taxi"
bash /root/mojolearn-pass31/tools/cell_ab_job2.sh nvidia treeshap "x_trees rf" "tree-shap" "" MOJOLEARN_XTREES_SHAP_TASKS=1 "istella taxi"
