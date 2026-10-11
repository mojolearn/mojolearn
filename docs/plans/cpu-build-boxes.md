# CPU build boxes: one way in (Andrew, 2026-10-11)

`tools/do_cpu_leg.sh` is the ONLY way to rent DigitalOcean CPU build boxes (the cheap fast-CPU cross-compile droplets).
No ad-hoc API calls, no evidence-folder scripts. Four droplets created by ad-hoc calls on 2026-10-07
(`mojolearn-targeted-ab-*-cpu-6fe3cfce38`) ran 4.5 days with nothing to kill them (~$115).

Every leg has a Mac dead-man (armed before the create), an on-droplet systemd self-destruct, a hard lease
(default 240 min, max 720), tag `mojolearn-cpu-leg` and the deadline in its name. At most 4 legs.

    tools/do_cpu_leg.sh up build1 240      # rent (THE BILL STARTS HERE)
    tools/do_cpu_leg.sh extend build1 120  # both deadlines -> now + 120 min
    tools/do_cpu_leg.sh down build1        # delete, verify 404, then drop state
    tools/do_cpu_leg.sh status
    tools/do_cpu_leg.sh sweep [--delete]   # untracked droplets; --delete removes ours older than 6 h
    tools/do_cpu_leg.sh up build1 --dry-run

Cron (install by hand):

    */30 * * * * /bin/bash $HOME/CascadeProjects/mojolearn/tools/do_cpu_leg.sh sweep --delete >> $HOME/mojolearn-evidence/do-cpu-leg/sweep.log 2>&1

Put a DigitalOcean custom-scoped token (droplet read + delete) in `~/.mojolearn_do_selfkill_token` so the on-droplet
self-destruct never holds the full account token. Test: `bash tools/tests/test_do_cpu_leg.sh` (API mocked, no network).
