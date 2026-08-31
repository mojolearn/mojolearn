"""Entry point for the radius-neighbours surface checks."""

from neighbors.mojo_only.radius_check import (
    check_radius_neighbors_matches_host,
    check_radius_neighbors_reach_by_sabotage,
    check_radius_neighbors_refuses_short_allocation,
    check_radius_neighbors_squared_arm,
)


def main() raises:
    check_radius_neighbors_matches_host()
    check_radius_neighbors_squared_arm()
    check_radius_neighbors_reach_by_sabotage()
    check_radius_neighbors_refuses_short_allocation()
