#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Full SSH for plain ROCm development images, using RunPod's injected key.
# https://docs.runpod.io/pods/configuration/use-ssh
# This is embedded in the create request and runs inside the container.
set -euo pipefail

: "${PUBLIC_KEY:?RunPod did not inject PUBLIC_KEY}"
if [ ! -x /usr/sbin/sshd ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 update
    apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 install -y --no-install-recommends openssh-server
fi
# Package installation may start a daemon with the image defaults. Stop it
# before binding port 22 with the explicit public-key-only configuration.
if [ -x /etc/init.d/ssh ]; then
    /etc/init.d/ssh stop || true
fi
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
umask 077
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
unset PUBLIC_KEY
ssh-keygen -A
# Ignore image-specific sshd configuration; every permitted authentication
# method is explicit. Internal SFTP supports the runner's artifact transfer.
exec /usr/sbin/sshd -D -e -f /dev/null \
    -o PermitRootLogin=prohibit-password \
    -o PasswordAuthentication=no \
    -o KbdInteractiveAuthentication=no \
    -o PubkeyAuthentication=yes \
    -o AuthenticationMethods=publickey \
    -o AuthorizedKeysFile=/root/.ssh/authorized_keys \
    -o AllowUsers=root \
    -o UsePAM=yes \
    -o 'Subsystem=sftp internal-sftp'
