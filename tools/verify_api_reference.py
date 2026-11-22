#!/usr/bin/env python3
"""Generate one shared reference and retain actual per-device validation."""
import argparse
import base64
import json
import textwrap
import zlib
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metal',type=Path)
    parser.add_argument('--cuda',type=Path,required=True)
    parser.add_argument('--hip',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    from mojolearn._verify_api_runner import admit
    data=admit({v:json.loads(getattr(args,v).read_text()) for v in ('metal','cuda','hip') if getattr(args,v)})
    payload=base64.b85encode(zlib.compress(json.dumps(data,sort_keys=True,separators=(',',':')).encode(),9)).decode()
    chunks='\n'.join('    '+repr(s) for s in textwrap.wrap(payload,100,break_on_hyphens=False,drop_whitespace=False))
    args.out.write_text('"""Generated endpoint reference; DATA records the actual validated vendors."""\n'
        'import base64 as _base64\nimport json as _json\nimport zlib as _zlib\n'
        '_PAYLOAD = (\n'+chunks+'\n)\n'
        'DATA = _json.loads(_zlib.decompress(_base64.b85decode(_PAYLOAD)))\ndel _PAYLOAD\n')
    print(json.dumps({'endpoints':len(data['witness']['expected']),
        'validated_vendors':data['validated_vendors'],'untested_vendors':data['untested_vendors']}))


if __name__=='__main__':main()
