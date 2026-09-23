import sys, struct, hashlib, os, re
def extract(path):
    d = open(path,'rb').read()
    out = []
    i = 1  # skip host ELF at 0
    while True:
        i = d.find(b'\x7fELF\x02\x01', i)
        if i < 0: break
        try:
            e_machine = struct.unpack_from('<H', d, i+18)[0]
            e_shoff = struct.unpack_from('<Q', d, i+0x28)[0]
            e_shentsize, e_shnum = struct.unpack_from('<HH', d, i+0x3A)
            size = e_shoff + e_shentsize*e_shnum
            if e_machine == 224 and 0 < size < len(d)-i:
                blob = d[i:i+size]
                # symbol names: find kernel names in strtab (".kd" suffix)
                names = sorted(set(m.decode() for m in re.findall(rb'([A-Za-z0-9_$.:\[\]<>,() -]{6,})\.kd\x00', blob)))
                out.append((i, size, hashlib.sha256(blob).hexdigest()[:16], names, blob))
                i += size; continue
        except Exception as ex:
            pass
        i += 4
    return out
if __name__ == '__main__':
    a = extract(sys.argv[1]); b = extract(sys.argv[2])
    print(len(a), len(b))
    for k,(x,y) in enumerate(zip(a,b)):
        same = x[2]==y[2]
        if not same or '-v' in sys.argv:
            print(k, 'SAME' if same else 'DIFF', x[1], y[1], x[3][:3])
        if not same and len(sys.argv)>3 and sys.argv[3] != '-v':
            open(f"{sys.argv[3]}/a_{k}.hsaco",'wb').write(x[4]); open(f"{sys.argv[3]}/b_{k}.hsaco",'wb').write(y[4])
