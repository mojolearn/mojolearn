#!/bin/sh
exec sh "$(dirname -- "$0")/build_host_family.sh" runtime_classical "$@"
