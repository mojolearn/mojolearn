"""Entry point for the brute-force k-NN checks."""

from neighbors.mojo_only.knn_check import check_knn, check_knn_reach_by_sabotage


def main() raises:
    check_knn()
    check_knn_reach_by_sabotage()
