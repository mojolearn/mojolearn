# SPDX-License-Identifier: Apache-2.0
# The same exact-integer target count used by the public Samba boundary.


def samba_target_count(address: Int, rows: Int) raises -> Int:
    if address == 0 or rows < 1:
        raise Error("Samba: target count requires a non-null buffer and positive rows")
    # The caller retains its admitted Int32 target buffer through this call.
    # Count directly: the old fill/compare/reduce path allocated two temporary
    # lists and passed their raw integer addresses without retaining the owners.
    # The installed optimizer-state verifier exposed negative target counts.
    # One exact-integer pass needs neither scratch storage nor a mask reduction.
    var targets = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=address)
    var count = 0
    for i in range(rows):
        if targets[i] != -100:
            count += 1
    return count
