#!/usr/bin/env bash
# tools/hotaisle_leg.sh. ONE GUARDED HOT AISLE AMD LEG (MI300X). SKELETON ONLY.
#
#   STATUS 2026-09-11: NOT BUILT. NEVER DRY-RUN. NEVER RUN. THIS FILE RENTS
#   NOTHING AND CALLS NO API: every mode below exits 2 with "NOT BUILT".
#   The lane was wound down after the API was verified and before the runner
#   code was written. What is here is the verified API surface, the agreed
#   design and the guard list, so the next lane builds from facts, not from
#   memory. Do not delete the refusal at the bottom until every item in
#   "UNFINISHED" is done and the RUN OWED sequence has started.
#
# INTENDED INTERFACE (mirrors tools/do_extra_leg.sh; none of it works yet)
#   MOJOLEARN_HOTAISLE_TEAM=<team handle> \
#   MOJOLEARN_GEMM_LEG_EXTRA=<body.sh> \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<stamp>-amd-mi300x-hotaisle-<lane> \
#   MOJOLEARN_HOTAISLE_EXTRA_ENV='MOJOLEARN_X=1 ...' \
#   bash tools/hotaisle_leg.sh [--dry-run | --probe | --rent [--watchdog-test [--fuse-minutes N]]]
#                              [--minutes N] [--skip-gates] [--allow-concurrent] [--gpu MI300X]
#   bash tools/hotaisle_leg.sh reap <vm-name>   DELETE ?force=true now, verify gone
#   bash tools/hotaisle_leg.sh leases           lease files and minutes left, no API
# ----
#
# RUN OWED, IN THIS ORDER, NONE STARTED
#   0. build the runner (UNFINISHED below), then `bash -n` and a code review.
#   1. `--dry-run` from `git worktree add --detach` (no key read, no API call).
#   2. `--probe` (free GETs only: teams, balance, offerings with price, VMs,
#      ssh keys).
#   3. `--rent --watchdog-test`: one short MI300X VM, trivial body (device
#      probe only). The on-box watchdog fires early and must DELETE the VM
#      through the API; the Mac issues no delete and verifies the VM is really
#      GONE (GET 404 AND absent from a 200 team listing), not merely stopped.
#      Red means no real leg is rented until it is green.
#   4. the first `--rent` with a tiny body (for example `pixi run mojo
#      --version` plus `rocminfo | grep gfx`).
#   Opponent rows from this box are a NEW TUPLE (Hot Aisle MI300X). Never mix
#   them with MI325X (DigitalOcean) or H100 (RunPod) rows.
#
# THE HOT AISLE API, VERIFIED 2026-09-11 (no call was made; read from
# https://admin.hotaisle.app/api/docs/swagger.json via WebFetch, and the raw
# copy in github.com/hotaisle/hotaisle-cli: swagger.json, client/client.go,
# client/models.go, client/virtual_machine_service.go,
# cmd/cli/command_virtual_machine.go; github.com/hotaisle/cloud-init-templates
# README.md and vllm-docker.yaml)
#   * swagger 2.0, host admin.hotaisle.app, scheme https, basePath /api/.
#     The CLI's DefaultBaseURL is https://admin.hotaisle.app/api.
#   * AUTH: securityDefinitions.token = apiKey, in header, name Authorization,
#     "enter the word 'Token', a space, and then paste in your token". The
#     templates README spells it `-H "Authorization: Token ${API_TOKEN}"`.
#     Example token shape in the spec: hex, a dot, hex.
#   * GET  /teams/  -> [UserTeam]: handle (required), name (required),
#     description, maximum_virtual_machines, maximum_bare_metal_servers,
#     roles and effective_roles (each of owner|purchaser|operator|user; the
#     effective ones are what THIS API KEY may do), invitation (bool; skip
#     true). Create and DELETE need team role operator.
#   * GET  /teams/{team}/balance/ -> BalanceInfo: available_balance (int
#     cents, required), hourly_rate (cents per hour, required; int in the CLI
#     copy, double in the live spec), virtual_machine_count and
#     bare_metal_server_count (required), minimum_balance (only when nonzero),
#     estimated_runout_time. (Coordinator: billing is per minute; not stated
#     in the spec.)
#   * GET  /teams/{team}/virtual_machines/available/ ->
#     [AvailableVirtualMachineTypes]: Quantity and MinimumReservationMinutes
#     (required; spec example 30), OnDemandPrice (US cents per hour), Specs
#     (VirtualMachineSpecs: cpu_cores, ram_capacity bytes, disk_capacity bytes,
#     cpus, gpus [{count (required), manufacturer, model}], example model
#     "MI300X").
#   * POST /teams/{team}/virtual_machines/  body VMProvisionRequest =
#     VirtualMachineSpecs + optional user_data_url. NO NAME FIELD: the server
#     assigns the VM name. Responses: CLI copy 200, live spec 201, both with
#     VirtualMachineDetails; 402 insufficient balance; 404 no matching VM;
#     401 (CLI copy) or 403 (live) for the team VM limit; the live spec adds
#     428 "Team has no accepted user-role member with an SSH key" and a
#     `force` query flag that overrides only that 428. "Provisioning will
#     continue in the background if the HTTP request is canceled", so a
#     timed-out create can still leave a billing VM. Without user_data_url the
#     VM is "available immediately"; with it the VM is rebuilt (slower).
#   * GET  /teams/{team}/virtual_machines/ -> [VirtualMachineDetails]
#     (live WebFetch summary said [VirtualMachine]).
#   * GET  /teams/{team}/virtual_machines/{vm}/ -> VirtualMachineDetails =
#     VirtualMachine {name, ip_address (required), description, ssh_access
#     ExternalService {ip_address, port (required), dns_name}} + Specs. The
#     live summary also listed deployment_id (uuid) as required and {vm} as
#     "name or deployment ID"; the CLI copy has neither. UNRESOLVED.
#   * GET  /teams/{team}/virtual_machines/{vm}/state/ -> {state, host}, state
#     values "running", "shut off", "paused", etc.
#   * PATCH /teams/{team}/virtual_machines/{vm}/  {description} -> 204. The
#     only way to MARK a VM as a mojolearn leg, since the name is not ours.
#   * DELETE /teams/{team}/virtual_machines/{vm}/ -> 204. CLI copy: query
#     `force` boolean, "the VM can be deleted even if the minimum reservation
#     time has not been met. However, the minimum charge ... will NOT be
#     refunded", and "This request will not return until the reset is
#     complete, but the reset will continue if the request is canceled".
#     The live WebFetch summaries disagreed with each other on `force`;
#     the watchdog and teardown must send ?force=true regardless, or a leg
#     shorter than MinimumReservationMinutes cannot be ended.
#   * The CLI labels `vm stop` "Continues billing" and `vm delete` "Ends
#     billing". Only DELETE stops the bill (same lesson as RunPod,
#     memory rented-gpus-self-expire).
#   * Also present, not needed: start, stop, shutdown, reboot, hard-reset,
#     rebuild (CLI copy) or reset (live), console, /user/ssh_keys/ (GET lists
#     {type, public_key, fingerprint "SHA256:...", comment}), /user/api_keys/
#     (a key can be restricted per team with roles).
#   * Error bodies are plain strings, not JSON.
#
# THE BOX, VERIFIED FROM THE SPEC AND THE TEMPLATES (not from a VM)
#   * SSH USER IS `hotaisle`. Spec tag "ssh": keys of any team member with the
#     `user` role may log in as `hotaisle`; sshd fetches them live from the
#     API (/etc/ssh/sshd_config.d/70-hotaisle-auth-keys.conf, cache in
#     ~/.ssh/hotaisle_managed_keys). Port comes from ssh_access.port.
#   * ufw is enabled, port 22 only (templates README).
#   * The template runs docker `rocm/vllm` with --device /dev/kfd and
#     /dev/dri, so docker and the amdgpu kernel driver are present. WHETHER
#     HOST ROCm USERLAND (rocminfo, rocm-smi, /opt/rocm) SHIPS IS UNVERIFIED.
#   * Passwordless sudo for `hotaisle` is UNVERIFIED. The body contract uses
#     /root paths, so the plan is to run every remote script as
#     `sudo -n -H sh -s` with the script on stdin, and refuse (delete the VM
#     unused) when `sudo -n true` fails.
#
# THE DESIGN (agreed; to be built)
#   * Key: MOJOLEARN_HOTAISLE_KEY_FILE, default $HOME/.mojolearn_hotaisle_key,
#     mode 600, one line, outside the repository, not tracked. Read once by
#     the builtin `read`, characters limited to [A-Za-z0-9._-], written by the
#     builtin `printf` into a 0600 curl config (`header = "Authorization:
#     Token <key>"`) read with `curl -K`. Never exported, never in an argv.
#     The same config reaches the VM on ssh STDIN for the watchdog. The
#     local `ps` and the VM's `ps` are searched for the key with
#     `grep -F -f <pattern file>` into key_in_ps.txt.
#   * Team: MOJOLEARN_HOTAISLE_TEAM, else the single non-invitation team
#     from GET /teams/; several teams refuse. effective_roles must include
#     operator, or the watchdog's DELETE would 403.
#   * Modes: dry run by default; --probe free GETs only; --rent bills.
#   * GPU: MI300X only for now; one GPU; the smallest Quantity>0 offering
#     with exactly one MI300X; its Specs copied into create_request.json;
#     price, MinimumReservationMinutes and balance printed before the create.
#     Refuse when available_balance is below price times
#     max(lease, minimum reservation) plus minimum_balance.
#   * Pre-flight refuses when the team already has a VM whose description
#     starts with the leg marker, or a VM with no description (possibly a
#     leg whose PATCH never landed), unless --allow-concurrent. The GET
#     /user/ssh_keys/ list must contain this Mac's ~/.ssh/id_ed25519.pub
#     fingerprint. No DigitalOcean GPU lock (Hot Aisle is not that account).
#   * LOCAL DEAD-MAN, detached, armed BEFORE the create: DELETE ?force=true of
#     the recorded VM name, plus any VM absent from the pre-create name
#     snapshot whose description is empty or carries this leg's nonce (the
#     unreadable-create case, since the name is server assigned).
#   * Create, then PATCH description "mojolearn-leg nonce=<nonce> ...". An
#     unreadable create response is adopted by the snapshot diff.
#   * ON-BOX WATCHDOG, armed BEFORE any work: key config on stdin to
#     /root/.mojolearn-hotaisle.curlrc (0600), watchdog started with
#     nohup setsid, sleeps to the lease end, then DELETE ?force=true of its
#     own VM, retried. Verified by pid alive, VM name baked in, a GET of its
#     own VM from the box returning 200, and a SECOND ssh session finding the
#     pid still alive (sudo use_pty may SIGHUP children). Unverifiable means
#     delete unused. Lease file bench/results/hotaisle_leases/<vm>.lease.
#   * GPU arch: device probe records rocminfo gfx names and GPU agent count,
#     KFD topology gfx_target_version raw values, /dev/kfd, amdgpu module,
#     ROCm version, rocm-smi product name. One gfx agent required; rocminfo
#     required for a real leg (/dev/dri alone is not AMD evidence). Unset
#     MOJOLEARN_GPU_ARCHS takes the box value; a set one that differs
#     refuses before the source ships. The remote body rechecks it.
#   * Body contract: tools/do_extra_leg.sh's remote body with
#     provider=hotaisle, rocm-smi as @SMI@, MOJOLEARN_TARGET_COLUMN=amd,
#     MOJOLEARN_HOTAISLE_EXTRA_ENV validated as MOJOLEARN_* or MODULAR_*
#     names with values in [A-Za-z0-9_.,:/=-], refusing MOJOLEARN_HOTAISLE_*,
#     MOJOLEARN_DO_*, MOJOLEARN_RUNPOD_*, MOJOLEARN_GEMM_LEG_*,
#     MOJOLEARN_GPU_ARCHS and MOJOLEARN_TARGET_COLUMN. Source is git archive
#     of the pinned clean commit, streamed over ssh stdin (no scp), sha256
#     checked on the box.
#   * Teardown on EXIT: DELETE ?force=true (long max-time, the call blocks),
#     then poll until GET 404 AND absent from a 200 listing; only then cancel
#     the dead-man and remove the lease. Otherwise a banner, both guards left
#     armed.
#   * Evidence: everything do_extra_leg.sh writes (leg.txt, teardown.txt,
#     deadman.txt, extra_body.sh, extra_env.sh, bundle_files.txt,
#     remote_body.sh, remote/), plus create_request.json,
#     create_response.json (key redacted), offering.json,
#     balance_before.json, vm_states.txt, device_probe.txt, key_in_ps.txt,
#     watchdog_test.txt.
#
# UNFINISHED (everything executable)
#   the argument parser and modes; key and team loading; probe; offering
#   pick and balance guard; pre-flight; local dead-man; create, adoption and
#   PATCH; wait for running; ssh settle and sudo check; on-box watchdog arm
#   and verification; lease files and reap/leases; device probe and arch
#   decision; bundle, upload, unpack; remote body and start wrapper; poll;
#   fetch; teardown and verification; the --watchdog-test flow; the dry run.
#
# RISKS ONLY A REAL RUN CAN SETTLE
#   whether DELETE needs force=true in practice and how long it blocks;
#   whether a watchdog on the VM survives the arming ssh session under sudo;
#   whether its DELETE completes when the VM it runs on is torn down mid
#   request; whether host ROCm userland and passwordless sudo exist; the VM
#   name format and whether {vm} also accepts deployment_id; the create
#   status code (200 or 201) and how long the POST blocks; the MI300X gfx
#   name; per-minute billing versus MinimumReservationMinutes; that the API
#   key placed on the VM can create VMs (mint a key restricted to this team,
#   operator role, and rotate it if key_in_ps.txt ever shows it).
set -uo pipefail

echo "tools/hotaisle_leg.sh: NOT BUILT. This is a skeleton (see its header)." >&2
echo "  It has never been dry-run and it rents nothing. Args ignored: $*" >&2
exit 2
