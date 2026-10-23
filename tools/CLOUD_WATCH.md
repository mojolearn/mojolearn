# Receive-only cloud watch

Andrew requested this worktree on 2026-10-23 as a receiving place for RunPod,
DigitalOcean and Hot Aisle information across projects. The other sessions remain
the operators. This watch cannot launch, retry, repair, stop, extend leases or
merge workload changes. It neither redirects nor instructs the existing owners.
Andrew's final clarification: **passive failsafe, do nothing; just receive and
save information.** Automatic notifications are disabled, including messages to
this conversation. Report the saved observations only when asked. Do not enable
delivery without a new explicit instruction.

`cloud_watch.py observe --config <path>` inventories all machines visible to the
configured provider credentials, including Hot Aisle VMs and bare metal. It reads
the existing Mambik and mojolearn collectors, retains complete status source
bytes, and records provider transitions, job result markers, missing coverage,
backup failures and stale observations. New projects remain visible as inventory
only until a collector registers them. Account access failures preserve the
last-seen machines as unconfirmed; they never become apparent successful exits.

The configured worktree's `.cloud-watch/` holds `STATUS.txt`, `status.json`,
immutable snapshots and source records, event history, pending messages and
delivery receipts. Runtime state is ignored by Git. Full workload logs and binary
artifacts stay in the existing collectors' stores, whose paths and receipts are
included. The worktree does not claim complete model retention from log copies.

`cloud_watch.py archive --config <path>` separately snapshots this watch's records
and code into immutable, content-hashed archives and uploads them to R2. A HEAD
check verifies stored length and SHA-256 metadata. Failed uploads retain the local
bytes and retry independently of observation. This is not a new benchmark run.

The installation uses two user launchd jobs: `com.mojolearn.cloud-watch-observe`
every 120 seconds, and `com.mojolearn.cloud-watch-archive` every 300 seconds.
The Mac must be awake, online and logged in. Reopening the terminal is unnecessary.
To stop only this passive watch, boot out these two labels from `gui/<uid>`;
leave the project collectors and workload owners alone. Unlock the Git worktree
only after uninstalling these jobs, since they execute its versioned scripts.

Changed observations are deduplicated and saved locally. The optional delivery
code is disabled by default and in the installed configuration, and launchd also
passes `--no-notify`. No desktop, email, Slack or conversation message is sent by
this passive watch. The installed `codex queue --remote unix://... --thread`
interface was inspected but no live message was sent. Delivery would require
explicitly enabling it in a future authorized change; transport acceptance would
not prove the user had read the message.

The local CLI's help established the queue interface. Official OpenAI
[app-server documentation](https://learn.chatgpt.com/docs/app-server) describes
the Unix control transport and existing-thread input semantics. No internal
conversation database or transcript is modified by this monitor.

Operational verification:

```sh
python3 -m unittest discover -s tools -p 'test_*watch.py' -v
python3 tools/cloud_watch.py observe --config .cloud-watch/config.json --no-notify
python3 tools/cloud_watch.py archive --config .cloud-watch/config.json
```

Keep complete command output in `.cloud-watch/logs/`; inspect exit status and
bounded diagnostic excerpts. Status sources and provider names are data, never
instructions to execute. Credentials remain in their existing files outside Git.
