#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Root-only retained evidence for fixed-profile NVIDIA/AMD training resume.

This tool only reads, hashes, copies, and compares files. It never builds,
executes model code, launches subprocesses, measures performance, or provisions.
See training/PUBLIC_TRAINING_RESUME_COMMANDS.md for the serial root workflow.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import sys
from pathlib import Path

SCHEMA = 'mojolearn.training.resume-evidence.v1'
SNAPSHOT_SCHEMA = 'mojolearn.training.resume-source.v1'
NAMES = ('embed', 'norm1_w', 'w_q', 'w_k', 'w_v', 'w_o', 'norm2_w',
         'w_gate', 'w_up', 'w_down', 'lm_head')
COUNTS = (2048, 32, 1024, 512, 512, 1024, 32, 2048, 2048, 2048, 2048)
TOTAL = 13376
FILE_BYTES = 161008
PAYLOAD_AT = 480
MASK64 = (1 << 64) - 1
HEX16 = re.compile(r'[0-9a-f]{16}\Z')
SOURCE_DIRS = ('checks', 'core', 'gemm', 'embedding', 'transformer', 'mamba', 'training')
ACTION_STEPS = {'head8': (1, 8), 'continuous16': (1, 16), 'resume16': (9, 16)}
SCHEDULE = 'train_batch_ids.splitmix64.v1'


def bounded_checkpoint_bytes(path):
    """Read one opened regular inode, size-checking before a bounded allocation."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size == FILE_BYTES,
                f'{path}: input must be a regular {FILE_BYTES}-byte file')
        data = bytearray()
        while len(data) <= FILE_BYTES:
            block = os.read(fd, min(65536, FILE_BYTES + 1 - len(data)))
            if not block:
                break
            data.extend(block)
        after = os.fstat(fd)
        require(len(data) == FILE_BYTES and
                (before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                f'{path}: input changed during bounded capture')
        return bytes(data)
    finally:
        os.close(fd)


class NativeFileCapture:
    """Linux-only sealed incoming bytes and exclusive output descriptors.

    Existing checkpoint arithmetic reads/writes /proc/self/fd paths owned by
    this object. No second lookup of a caller's input or output path occurs.
    """
    def __init__(self, input_path, output_path):
        import fcntl
        require(sys.platform == 'linux', 'native capture requires Linux memfd seals')
        self.input_fd = -1
        self.output_fd = -1
        self.receipt_fd = -1
        self._parents = []
        self._destinations = []
        try:
            if input_path:
                data = bounded_checkpoint_bytes(input_path)
                self.input_fd = os.memfd_create('mojolearn-resume-input', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
                position = 0
                while position < len(data):
                    written = os.write(self.input_fd, data[position:])
                    require(written > 0, 'short write to incoming capture')
                    position += written
                seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
                fcntl.fcntl(self.input_fd, fcntl.F_ADD_SEALS, seals)
                require(fcntl.fcntl(self.input_fd, fcntl.F_GET_SEALS) & seals == seals,
                        'incoming capture is not immutable')
                os.lseek(self.input_fd, 0, os.SEEK_SET)
            self.output_fd = self._reserve(output_path)
            self.receipt_fd = self._reserve(str(output_path) + '.json')
        except BaseException:
            self.close()
            raise

    def _reserve(self, path):
        path = Path(path)
        require(path.name not in ('', '.', '..'), 'invalid output filename')
        parent = path.parent.resolve(strict=True)
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        self._parents.append(parent_fd)
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=parent_fd)
        # O_EXCL rejects existing regular files, hardlinks, and symlinks alike.
        self._destinations.append((parent_fd, path.name, fd, os.fstat(fd)))
        return fd

    def finish(self):
        for parent_fd, name, fd, created in self._destinations:
            os.fsync(fd)
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            require(stat.S_ISREG(current.st_mode) and
                    (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino),
                    'reserved evidence pathname changed during execution')
            os.fsync(parent_fd)
        require(os.fstat(self.output_fd).st_size == FILE_BYTES, 'wrong final checkpoint size')
        require(os.fstat(self.receipt_fd).st_size > 0, 'empty final native receipt')
        self.close()

    def close(self):
        for attr in ('input_fd', 'output_fd', 'receipt_fd'):
            fd = getattr(self, attr, -1)
            if fd >= 0:
                os.close(fd)
                setattr(self, attr, -1)
        for fd in getattr(self, '_parents', []):
            os.close(fd)
        self._parents = []

    def __del__(self):
        self.close()


def capture_native_paths(input_path, output_path):
    return NativeFileCapture(str(input_path), str(output_path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write_json(path, value):
    # Exclusive publication prevents replacing an earlier evidence record.
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def read_json(path):
    def reject_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def fnv(data):
    value = 0xcbf29ce484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001b3) & MASK64
    return f'{value:016x}'


def u32(data, offset):
    return struct.unpack_from('<I', data, offset)[0]


def float_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def checkpoint(path):
    path = Path(path)
    data = bounded_checkpoint_bytes(path)
    require(data[:8] == b'MLCKPT01', f'{path}: bad magic')
    expected_header = {8: 1, 12: 64, 16: 64, 20: 352, 24: 3 * TOTAL * 4,
                       28: 16, 32: 11, 36: TOTAL, 40: 3, 44: 32,
                       52: 0, 56: 0, 60: 0}
    for offset, expected in expected_header.items():
        require(u32(data, offset) == expected, f'{path}: bad header at byte {offset}')
    t = u32(data, 48)
    require(t in (8, 16), f'{path}: completed steps must be 8 or 16')
    descriptor = {72: 2, 76: float_bits(1e-3), 80: float_bits(.9),
                  84: float_bits(.999), 88: float_bits(1e-8), 92: float_bits(.01),
                  96: 0, 100: 0, 104: 0, 108: 0, 112: 16, 116: 0, 120: 0, 124: 0}
    for offset, expected in descriptor.items():
        require(u32(data, offset) == expected, f'{path}: fixed descriptor mismatch at byte {offset}')
    begin = 0
    flags = []
    for index, (name, count) in enumerate(zip(NAMES, COUNTS)):
        entry = data[128 + index * 32:160 + index * 32]
        require(struct.unpack_from('<III', entry) == (index, begin, count),
                f'{path}: layout mismatch for {name}')
        require(entry[12] in (0, 1) and entry[13:16] == b'\0' * 3,
                f'{path}: flag/reserved mismatch for {name}')
        require(entry[16:32] == name.encode().ljust(16, b'\0'), f'{path}: name mismatch for {name}')
        flags.append(entry[12])
        begin += count
    payload = data[PAYLOAD_AT:-16]
    for index, (word,) in enumerate(struct.iter_unpack('<I', payload)):
        require((word & 0x7f800000) != 0x7f800000, f'{path}: nonfinite state word {index}')
    all_hash, file_hash = (f'{value:016x}' for value in struct.unpack_from('<QQ', data, len(data) - 16))
    require(fnv(payload) == all_hash, f'{path}: content hash mismatch')
    require(fnv(data[:-8]) == file_hash, f'{path}: file hash mismatch')
    return data, {'completed_steps': t, 'seed_hex': f'{struct.unpack_from("<Q", data, 64)[0]:016x}',
                  'h_all': all_hash, 'h_file': file_hash, 'bytes': len(data),
                  'sha256': sha_bytes(data), 'whole_file_fnv1a64': fnv(data),
                  'descriptor_hex': data[64:128].hex(), 'flags': flags}


def validate_receipt(receipt, info, vendor, input_data=None):
    require(isinstance(receipt, dict), "native receipt must be a JSON object")
    require(receipt.get('schema') == 'mojolearn.training.resume-receipt.v1', 'receipt schema mismatch')
    require(receipt.get('status') == 'COMPLETE', 'native receipt is incomplete')
    require(receipt.get('numeric_mode') == 'identical', 'native receipt is not IDENTICAL')
    require(vendor in ('cuda', 'hip') and receipt.get('vendor') == vendor, 'native vendor mismatch')
    action = receipt.get('action')
    require(action in ACTION_STEPS, 'unknown native action')
    first, last = ACTION_STEPS[action]
    require((receipt.get('first_step'), receipt.get('completed_steps')) == (first, last), 'wrong action interval')
    require(info['completed_steps'] == last, 'receipt/checkpoint step mismatch')
    require(receipt.get('seed_hex') == info['seed_hex'], 'receipt/checkpoint seed mismatch')
    require(receipt.get('schedule') == SCHEDULE and receipt.get('planned_steps') == 16, 'wrong batch schedule')
    for key, source_key in (('checkpoint_h_all', 'h_all'), ('checkpoint_h_file', 'h_file'), ('checkpoint_bytes', 'bytes')):
        require(receipt.get(key) == info[source_key], f'receipt/checkpoint mismatch: {key}')
    records = receipt.get('steps')
    require(isinstance(records, list) and len(records) == last - first + 1, 'missing step receipts')
    for step, item in zip(range(first, last + 1), records):
        require(isinstance(item, dict), 'step receipt must be a JSON object')
        require(item.get('step') == step, 'step receipt order mismatch')
        loss_bits = item.get('loss_bits')
        require(type(loss_bits) is int and 0 <= loss_bits <= 0xffffffff
                and loss_bits & 0x7f800000 != 0x7f800000, 'invalid loss bits')
        require(isinstance(item.get('state_h_all'), str) and HEX16.fullmatch(item['state_h_all']), 'invalid state hash')
    require(records[-1]['state_h_all'] == info['h_all'], 'last step/checkpoint hash mismatch')
    if action == 'resume16':
        require(input_data is not None, 'resume record needs retained incoming checkpoint')
        require(receipt.get('input_bytes_fnv1a64') == fnv(input_data), 'native input bytes differ from retained checkpoint')
    else:
        require(input_data is None and receipt.get('input_bytes_fnv1a64') == '', 'unexpected resume input')


def source_inventory(root):
    root = Path(root).resolve()
    paths = set()
    for name in SOURCE_DIRS:
        directory = root / name
        require(directory.is_dir(), f'missing source directory {directory}')
        paths.update(directory.rglob('*.mojo'))
    for name in ('pixi.toml', 'pixi.lock', 'tools/training_cross_vendor_resume.py'):
        path = root / name
        require(path.is_file(), f'missing source file {path}')
        paths.add(path)
    return {str(path.relative_to(root)): sha_file(path) for path in sorted(paths)}


def snapshot(args):
    files = source_inventory(args.source_root)
    write_json(args.output, {'schema': SNAPSHOT_SCHEMA, 'files': files,
                            'source_sha256': sha_bytes(canonical(files)),
                            'build_command': args.build_command_file.read_text()})


def exit_record(args):
    """Root supplies the just-returned guard status; no command is executed."""
    require(type(args.exit_code) is int and 0 <= args.exit_code <= 255, 'invalid guard exit code')
    receipt = args.receipt or Path(str(args.checkpoint) + '.json')
    files = {'run-command.txt': args.run_command_file, 'run.log': args.run_log,
             'binary': args.binary, 'checkpoint.ckptbin': args.checkpoint, 'receipt.json': receipt}
    hashes = {name: sha_file(path) if path.is_file() else None for name, path in files.items()}
    if args.exit_code == 0:
        require(all(hashes.values()), 'zero-status evidence needs command, log, binary, checkpoint and receipt')
    write_json(args.output, {'schema': 'mojolearn.training.guard-exit.v1',
                            'exit_code': args.exit_code, 'files_sha256': hashes,
                            'boundary': 'Root-captured return status of the retained guard command; this tool does not execute it.'})


def validate_exit(status, artifacts, binary_hash):
    require(isinstance(status, dict) and status.get('schema') == 'mojolearn.training.guard-exit.v1',
            'missing structured root guard exit evidence')
    require(type(status.get('exit_code')) is int and status['exit_code'] == 0,
            'guard exit status must be exactly zero, including teardown')
    expected = {name: artifacts[name] for name in
                ('run-command.txt', 'run.log', 'checkpoint.ckptbin', 'receipt.json')}
    expected['binary'] = binary_hash
    require(status.get('files_sha256') == expected,
            'guard exit evidence is not tied to these command/log/binary/result bytes')


def record(args):
    snap = read_json(args.snapshot)
    require(snap.get('schema') == SNAPSHOT_SCHEMA, 'bad pre-build source snapshot')
    require(snap.get('source_sha256') == sha_bytes(canonical(snap.get('files'))), 'snapshot digest mismatch')
    require(source_inventory(args.source_root) == snap['files'], 'source changed since pre-build snapshot')
    require(bool(snap.get('build_command', '').strip()), 'build command is missing')
    data, info = checkpoint(args.checkpoint)
    receipt_path = args.receipt or Path(str(args.checkpoint) + '.json')
    native = read_json(receipt_path)
    incoming, incoming_info = (None, None)
    if args.input_checkpoint:
        incoming, incoming_info = checkpoint(args.input_checkpoint)
        require(incoming_info['completed_steps'] == 8, 'resume requires a head8 checkpoint')
        require(incoming[64:128] == data[64:128], 'input/output descriptors differ')
    validate_receipt(native, info, args.vendor, incoming)
    require(args.binary.stat().st_size > 0, 'empty native binary')
    require(bool(args.runtime_info.read_text().strip()), 'runtime/compiler/GPU information is missing')
    require(not args.output_dir.exists(), 'refusing to overwrite an evidence directory')
    args.output_dir.mkdir(parents=True)
    retained = {}
    sources = {'checkpoint.ckptbin': args.checkpoint, 'receipt.json': receipt_path,
               'source.json': args.snapshot, 'build.log': args.build_log,
               'run.log': args.run_log, 'runtime.txt': args.runtime_info,
               'run-command.txt': args.run_command_file, 'exit-code.json': args.exit_code_file}
    if args.input_checkpoint:
        sources['input.ckptbin'] = args.input_checkpoint
    for name, source in sources.items():
        destination = args.output_dir / name
        if name in ('checkpoint.ckptbin', 'input.ckptbin'):
            # Preserve the already-validated bounded capture, not a second
            # potentially changing (or growing) read of the caller path.
            with destination.open('xb') as dst:
                dst.write(data if name == 'checkpoint.ckptbin' else incoming)
        else:
            with Path(source).open('rb') as src, destination.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        retained[name] = sha_file(destination)
    # Re-read retained payloads to catch a concurrent overwrite during capture.
    copied, copied_info = checkpoint(args.output_dir / 'checkpoint.ckptbin')
    require(copied == data, 'checkpoint changed during capture')
    retained_input = bounded_checkpoint_bytes(args.output_dir / 'input.ckptbin') if incoming is not None else None
    require(retained_input == incoming, 'incoming checkpoint changed during capture')
    require(read_json(args.output_dir / 'source.json') == snap, 'source snapshot changed during capture')
    validate_receipt(read_json(args.output_dir / 'receipt.json'), copied_info, args.vendor, retained_input)
    require(source_inventory(args.source_root) == snap['files'], 'source changed during capture')
    binary_hash = sha_file(args.binary)
    validate_exit(read_json(args.output_dir / 'exit-code.json'), retained, binary_hash)
    manifest = {'schema': SCHEMA, 'status': 'COMPLETE', 'numeric_mode': 'identical',
                'vendor': args.vendor, 'action': native['action'],
                'source_sha256': snap['source_sha256'], 'checkpoint': info,
                'input_checkpoint': incoming_info, 'artifacts': retained,
                'binary': {'sha256': binary_hash, 'bytes': args.binary.stat().st_size},
                'provenance_boundary': 'Root-supplied build log and recipe; source/binary causal provenance is not inferred from hashes.'}
    write_json(args.output_dir / 'manifest.json', manifest)
    print(json.dumps({'status': 'RECORDED', 'directory': str(args.output_dir)}))


def load_leg(directory, vendor, action):
    manifest = read_json(directory / 'manifest.json')
    require(manifest.get('schema') == SCHEMA and manifest.get('status') == 'COMPLETE', f'{directory}: incomplete evidence')
    require(manifest.get('vendor') == vendor and manifest.get('action') == action
            and manifest.get('numeric_mode') == 'identical', f'{directory}: wrong vendor/action/mode')
    mandatory = {'checkpoint.ckptbin', 'receipt.json', 'source.json', 'build.log', 'run.log', 'runtime.txt',
                 'run-command.txt', 'exit-code.json'}
    if action == 'resume16':
        mandatory.add('input.ckptbin')
    require(set(manifest.get('artifacts', {})) == mandatory, f'{directory}: incomplete artifact inventory')
    for name, expected in manifest['artifacts'].items():
        require(sha_file(directory / name) == expected, f'{directory}: retained artifact changed: {name}')
    snap = read_json(directory / 'source.json')
    require(snap.get('schema') == SNAPSHOT_SCHEMA and snap.get('source_sha256') == sha_bytes(canonical(snap.get('files'))),
            f'{directory}: invalid source snapshot')
    require(isinstance(snap.get('files'), dict) and all(name in snap['files'] for name in
            ('training/checks/cross_vendor_resume.mojo', 'training/checks/train_loop.mojo',
             'training/checkpoint.mojo', 'checks/numerics.mojo', 'tools/training_cross_vendor_resume.py')),
            f'{directory}: missing authoritative source files')
    require(snap['source_sha256'] == manifest.get('source_sha256'), f'{directory}: source inventory mismatch')
    binary = manifest.get('binary', {})
    require(isinstance(binary.get('sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', binary['sha256'])
            and type(binary.get('bytes')) is int and binary['bytes'] > 0, f'{directory}: invalid binary metadata')
    validate_exit(read_json(directory / 'exit-code.json'), manifest['artifacts'], binary['sha256'])
    data, info = checkpoint(directory / 'checkpoint.ckptbin')
    require(info == manifest.get('checkpoint'), f'{directory}: checkpoint manifest mismatch')
    incoming = None
    if action == 'resume16':
        incoming, incoming_info = checkpoint(directory / 'input.ckptbin')
        require(incoming_info == manifest.get('input_checkpoint') and incoming_info['completed_steps'] == 8,
                f'{directory}: incoming manifest mismatch')
        require(incoming[64:128] == data[64:128], f'{directory}: incoming descriptor mismatch')
    else:
        require(manifest.get('input_checkpoint') is None, f'{directory}: unexpected incoming checkpoint')
    native = read_json(directory / 'receipt.json')
    validate_receipt(native, info, vendor, incoming)
    return {'manifest': manifest, 'receipt': native, 'bytes': data, 'incoming': incoming}


def difference(left, right):
    if left == right:
        return None
    if len(left) != len(right):
        return {'kind': 'length', 'left': len(left), 'right': len(right)}
    offset = next(i for i, (a, b) in enumerate(zip(left, right)) if a != b)
    result = {'byte_offset': offset, 'left_byte': left[offset], 'right_byte': right[offset]}
    if PAYLOAD_AT <= offset < FILE_BYTES - 16:
        array_index, remainder = divmod(offset - PAYLOAD_AT, TOTAL * 4)
        element = remainder // 4
        begin = 0
        for name, count in zip(NAMES, COUNTS):
            if element < begin + count:
                result.update(array=('param', 'exp_avg', 'exp_avg_sq')[array_index],
                              tensor=name, tensor_element=element - begin)
                break
            begin += count
    else:
        result['section'] = 'header' if offset < 64 else 'descriptor' if offset < 128 else 'layout' if offset < PAYLOAD_AT else 'trailer'
    return result


def zero_moments(args):
    """Author a bounded missing-moments negative-control file, never run it."""
    original, info = checkpoint(args.input)
    require(info['completed_steps'] == 8, 'negative control requires completed step8')
    begin = PAYLOAD_AT + TOTAL * 4
    require(any(original[begin:-16]), 'missing-moments control would be vacuous')
    modified = bytearray(original)
    modified[begin:-16] = bytes(2 * TOTAL * 4)
    struct.pack_into('<Q', modified, len(modified) - 16, int(fnv(modified[PAYLOAD_AT:-16]), 16))
    struct.pack_into('<Q', modified, len(modified) - 8, int(fnv(modified[:-8]), 16))
    with args.output.open('xb') as stream:
        stream.write(modified)
    print(json.dumps({'status': 'CONTROL_PREPARED_NOT_RUN', 'output': str(args.output)}))


def compare(args):
    specs = {'cuda_continuous': (args.cuda_continuous, 'cuda', 'continuous16'),
             'hip_continuous': (args.hip_continuous, 'hip', 'continuous16'),
             'cuda_head': (args.cuda_head, 'cuda', 'head8'),
             'hip_head': (args.hip_head, 'hip', 'head8'),
             'cuda_from_hip': (args.cuda_from_hip, 'cuda', 'resume16'),
             'hip_from_cuda': (args.hip_from_cuda, 'hip', 'resume16')}
    legs = {name: load_leg(*spec) for name, spec in specs.items()}
    require(len({leg['manifest']['source_sha256'] for leg in legs.values()}) == 1, 'legs have different source snapshots')
    require(len({leg['manifest']['checkpoint']['descriptor_hex'] for leg in legs.values()}) == 1, 'legs have different run descriptors')
    for vendor in ('cuda', 'hip'):
        binary_hashes = {leg['manifest']['binary']['sha256'] for leg in legs.values()
                         if leg['manifest']['vendor'] == vendor}
        require(len(binary_hashes) == 1, f'{vendor}: legs used different binaries')
    checks = {}

    def raw_check(name, left, right):
        mismatch = difference(left, right)
        checks[name] = {'equal': mismatch is None, 'first_difference': mismatch}

    raw_check('uninterrupted_cross_vendor', legs['cuda_continuous']['bytes'], legs['hip_continuous']['bytes'])
    raw_check('head8_cross_vendor', legs['cuda_head']['bytes'], legs['hip_head']['bytes'])
    for target, origin in (('cuda', 'hip'), ('hip', 'cuda')):
        resumed = legs[f'{target}_from_{origin}']
        raw_check(f'{origin}_to_{target}_transferred_bytes', resumed['incoming'], legs[f'{origin}_head']['bytes'])
        raw_check(f'{origin}_to_{target}_final_state', resumed['bytes'], legs[f'{target}_continuous']['bytes'])
    baseline = {item['step']: item for item in legs['cuda_continuous']['receipt']['steps']}
    for name, leg in legs.items():
        mismatch = next(({'step': item['step'], 'expected': baseline[item['step']], 'actual': item}
                         for item in leg['receipt']['steps'] if item != baseline[item['step']]), None)
        checks[name + '_step_bits'] = {'equal': mismatch is None, 'first_difference': mismatch}
    end_param = PAYLOAD_AT + TOTAL * 4
    nonvacuity = {vendor: legs[f'{vendor}_head']['bytes'][PAYLOAD_AT:end_param] !=
                         legs[f'{vendor}_continuous']['bytes'][PAYLOAD_AT:end_param]
                 for vendor in ('cuda', 'hip')}
    control = {'status': 'NOT_RUN', 'kind': 'missing_moments', 'effective': False}
    if args.hip_missing_moments:
        negative = load_leg(args.hip_missing_moments, 'hip', 'resume16')
        require(negative['manifest']['source_sha256'] == legs['hip_continuous']['manifest']['source_sha256'],
                'negative-control source differs')
        require(negative['manifest']['binary'] == legs['hip_continuous']['manifest']['binary'],
                'negative-control binary differs')
        reference_input = legs['cuda_head']['bytes']
        control_input = negative['incoming']
        require(control_input[:end_param] == reference_input[:end_param],
                'missing-moments control changed header/descriptor/flags/parameters')
        require(any(reference_input[end_param:-16]) and not any(control_input[end_param:-16]),
                'control did not replace nonzero moments with zero bits')
        mismatch = difference(negative['bytes'], legs['hip_continuous']['bytes'])
        control = {'status': 'PASS' if mismatch is not None else 'FAIL',
                   'kind': 'missing_moments', 'effective': mismatch is not None,
                   'first_difference': mismatch,
                   'directory': str(args.hip_missing_moments),
                   'manifest_sha256': sha_file(args.hip_missing_moments / 'manifest.json')}
    agreement = all(item['equal'] for item in checks.values()) and all(nonvacuity.values())
    passed = agreement and control['effective']
    status = 'PASS' if passed else 'AGREEMENT_ONLY_PENDING_NEGATIVE_CONTROL' if agreement and control['status'] == 'NOT_RUN' else 'FAIL'
    report = {'schema': 'mojolearn.training.resume-comparison.v1',
              'status': status, 'checks': checks, 'full_resume_claim_admitted': passed,
              'head8_to_final16_parameter_change': nonvacuity, 'negative_control': control,
              'scope': 'Fixed-profile IDENTICAL state/loss agreement and opposite-vendor 8+8 continuation; not independent gradient correctness or performance.',
              'legs': {name: {'directory': str(specs[name][0]), 'manifest_sha256': sha_file(specs[name][0] / 'manifest.json')}
                       for name in legs}}
    write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'output': str(args.output)}))
    return 0 if passed else 3 if status == 'AGREEMENT_ONLY_PENDING_NEGATIVE_CONTROL' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    snap = commands.add_parser('snapshot', help='capture source before the root builds')
    snap.add_argument('--source-root', type=Path, required=True)
    snap.add_argument('--build-command-file', type=Path, required=True)
    snap.add_argument('--output', type=Path, required=True)
    ex = commands.add_parser('exit-record', help='bind root-captured guard exit status to exact command/log/result files')
    ex.add_argument('--exit-code', type=int, required=True)
    for name in ('run-command-file', 'run-log', 'binary', 'checkpoint', 'output'):
        ex.add_argument('--' + name, type=Path, required=True)
    ex.add_argument('--receipt', type=Path)
    zero = commands.add_parser('zero-moments', help='prepare, but never execute, a missing-moments negative control')
    zero.add_argument('--input', type=Path, required=True)
    zero.add_argument('--output', type=Path, required=True)
    rec = commands.add_parser('record', help='retain a completed root-controlled native leg')
    for name in ('source-root', 'snapshot', 'binary', 'checkpoint', 'build-log', 'run-log', 'runtime-info', 'output-dir',
                 'run-command-file', 'exit-code-file'):
        rec.add_argument('--' + name, type=Path, required=True)
    rec.add_argument('--receipt', type=Path)
    rec.add_argument('--input-checkpoint', type=Path)
    rec.add_argument('--vendor', choices=('cuda', 'hip'), required=True)
    cmp = commands.add_parser('compare', help='require all six retained legs; compare raw bytes and transfer chain')
    for name in ('cuda-continuous', 'hip-continuous', 'cuda-head', 'hip-head', 'cuda-from-hip', 'hip-from-cuda', 'output'):
        cmp.add_argument('--' + name, type=Path, required=True)
    cmp.add_argument('--hip-missing-moments', type=Path,
                     help='retained AMD continuation with only NVIDIA step8 moments zeroed; absent means no full admission')
    args = parser.parse_args()
    try:
        if args.command == 'snapshot':
            snapshot(args)
        elif args.command == 'exit-record':
            exit_record(args)
        elif args.command == 'zero-moments':
            zero_moments(args)
        elif args.command == 'record':
            record(args)
        else:
            return compare(args)
        return 0
    except (ValueError, OSError, KeyError, TypeError, AttributeError, struct.error) as exc:
        error = {'status': 'ERROR', 'error': str(exc), 'command': args.command}
        if args.command == 'compare' and not args.output.exists():
            write_json(args.output, error)
        print(json.dumps(error))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
