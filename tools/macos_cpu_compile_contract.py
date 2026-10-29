"""CPU-only Mac linker identity read from the ORIGINAL retained Mach-O image.

build_host_family.sh uses Mojo/LLVM with apple-m1 and explicit platform_version,
not the Metal compiler. bincache.darwin_toolchain describes the Metal half of a
GPU binding and includes unrelated host OS/Xcode/Metal labels. This narrower
contract keeps the actual CPU compiler/package lock and link-emitted target,
SDK, linker version and dynamic dependency commands. It is a VERSION witness
for the linker/SDK, not a claim to possess their original executable/content
hashes. Runtime library loading and wheel assembly have separate requirements.

The field layout/constants are the supported Mach-O loader.h ABI. Read-only:
no compiler, current Xcode query, binary rewrite or inferred past labels.
"""
import re
import struct
from pathlib import Path

SCHEMA = 'mojolearn.macos-cpu-compile.v1'
HEX64 = re.compile('[0-9a-f]{64}')
DYLIB_COMMANDS = {0xC, 0x80000018, 0x8000001F, 0x80000023, 0x20}
# Standard non-dependency metadata commands observed/supported for 64-bit dylibs.
# Unknown future load commands require review instead of hiding a dependency.
METADATA_COMMANDS = {0x19,0x1B,0x1D,0x2,0x26,0x29,0x2A,0xB,0xD,
                     0x80000022,0x80000033,0x80000034}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def version(number):
    return [number >> 16, (number >> 8) & 255, number & 255]


def sdk_version(text):
    require(isinstance(text, str) and re.fullmatch(r'[0-9]+(?:\.[0-9]+){0,2}', text), 'invalid SDK version witness')
    parts = list(map(int, text.split('.')))
    return parts + [0] * (3-len(parts))


def read_image(path):
    with Path(path).open('rb') as source:
        header = source.read(32)
        require(len(header) == 32, 'truncated Mach-O header')
        magic, cpu, subtype, kind, count, size, _flags, reserved = struct.unpack('<8I', header)
        require((magic,cpu,subtype,kind,reserved) == (0xFEEDFACF,0x100000C,0,6,0),
                'CPU compile witness requires one little-endian ARM64/ALL dylib')
        require(0 < count <= 4096 and 8*count <= size <= 1<<20, 'invalid Mach-O load-command extent')
        commands = source.read(size)
    require(len(commands) == size, 'truncated Mach-O load commands')
    offset = 0
    builds, dependencies, rpaths = [], [], []
    for _ in range(count):
        require(offset+8 <= size, 'truncated Mach-O command header')
        command, length = struct.unpack_from('<2I', commands, offset)
        require(length >= 8 and length % 8 == 0 and offset+length <= size, 'invalid Mach-O command size')
        block = commands[offset:offset+length]
        if command == 0x32:  # LC_BUILD_VERSION
            require(length >= 24, 'truncated LC_BUILD_VERSION')
            platform, minimum, sdk, tools = struct.unpack_from('<4I', block, 8)
            require(length == 24+8*tools, 'invalid LC_BUILD_VERSION tool extent')
            builds.append(dict(platform=platform, minimum_os=version(minimum), sdk=version(sdk),
                               tools=[dict(tool=t, version=version(v)) for t,v in
                                      (struct.unpack_from('<2I',block,24+8*i) for i in range(tools))]))
        elif command == 0x8000001C:  # LC_RPATH, original link search path
            require(length >= 16, 'truncated LC_RPATH')
            start = struct.unpack_from('<I', block, 8)[0]
            require(12 <= start < length, 'invalid LC_RPATH offset')
            end = block.find(b'\0', start)
            require(end >= 0, 'unterminated LC_RPATH')
            try:
                rpaths.append(block[start:end].decode('utf-8'))
            except UnicodeDecodeError as error:
                raise ValueError('invalid LC_RPATH encoding') from error
        elif command in DYLIB_COMMANDS:
            require(length >= 24, 'truncated dylib dependency command')
            name_offset, _timestamp, current, compatibility = struct.unpack_from('<4I',block,8)
            require(24 <= name_offset < length, 'invalid dylib dependency name offset')
            end = block.find(b'\0',name_offset)
            require(end >= 0, 'unterminated dylib dependency name')
            try:
                name = block[name_offset:end].decode('utf-8')
            except UnicodeDecodeError as error:
                raise ValueError('invalid dylib dependency name encoding') from error
            dependencies.append(dict(command=command, name=name, current=version(current),
                                     compatibility=version(compatibility)))
        else:
            require(command in METADATA_COMMANDS, 'unrecognized Mach-O load command')
        offset += length
    require(offset == size and len(builds) == 1, 'missing, duplicate or trailing build commands')
    image = dict(cpu=cpu, subtype=subtype, filetype=kind, build=builds[0], dependencies=dependencies, rpaths=rpaths)
    validate_image(image)
    return image


def _version(value):
    return (isinstance(value,list) and len(value)==3 and all(type(x) is int for x in value)
            and 0 <= value[0] <= 65535 and all(0 <= x <= 255 for x in value[1:]))


def validate_image(image):
    require(isinstance(image,dict) and set(image)=={'cpu','subtype','filetype','build','dependencies','rpaths'},
            'incomplete CPU image witness')
    require(all(type(image[k]) is int for k in ('cpu','subtype','filetype')) and
            (image['cpu'],image['subtype'],image['filetype'])==(0x100000C,0,6), 'wrong CPU image target')
    build=image['build']
    require(isinstance(build,dict) and set(build)=={'platform','minimum_os','sdk','tools'} and
            type(build['platform']) is int and build['platform']==1 and
            _version(build['minimum_os']) and build['minimum_os']==[11,0,0] and
            _version(build['sdk']) and build['sdk'][0]>0, 'wrong CPU platform/minimum/SDK')
    require(isinstance(build['tools'],list) and len(build['tools'])==1 and
            isinstance(build['tools'][0],dict) and set(build['tools'][0])=={'tool','version'} and
            type(build['tools'][0]['tool']) is int and build['tools'][0]['tool']==3 and
            _version(build['tools'][0]['version']) and build['tools'][0]['version'][0]>0,
            'missing original linker version')
    require(isinstance(image['rpaths'],list) and image['rpaths'] and
            all(isinstance(p,str) and p.startswith(('/', '@loader_path', '@executable_path')) and
                '\0' not in p for p in image['rpaths']) and
            len(set(image['rpaths']))==len(image['rpaths']), 'missing or malformed original link search paths')
    rows=image['dependencies']
    require(isinstance(rows,list) and rows, 'missing dynamic dependency commands')
    names=set()
    for row in rows:
        require(isinstance(row,dict) and set(row)=={'command','name','current','compatibility'} and
                type(row['command']) is int and row['command'] in DYLIB_COMMANDS and isinstance(row['name'],str) and
                row['name'].startswith(('/', '@rpath/')) and '\0' not in row['name'] and
                _version(row['current']) and _version(row['compatibility']), 'malformed dynamic dependency witness')
        require(row['name'] not in names, 'duplicate dynamic dependency witness')
        names.add(row['name'])
    require('/usr/lib/libSystem.B.dylib' in names, 'missing CPU system library dependency')
    return image


def contract(compiler_sha256, pixi_lock_sha256, image):
    result=dict(schema=SCHEMA, target='osx-arm64', compiler_sha256=compiler_sha256,
                pixi_lock_sha256=pixi_lock_sha256, image=image)
    validate_contract(result)
    return result


def validate_contract(value):
    require(isinstance(value,dict) and set(value)=={'schema','target','compiler_sha256','pixi_lock_sha256','image'}
            and value['schema']==SCHEMA and value['target']=='osx-arm64', 'invalid Mac CPU compile contract')
    require(all(isinstance(value[k],str) and HEX64.fullmatch(value[k])
                for k in ('compiler_sha256','pixi_lock_sha256')), 'missing CPU compiler/package hash')
    validate_image(value['image'])
    return value
