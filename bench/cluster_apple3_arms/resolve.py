import sys, difflib, re
def resolve(path):
    lines=open(path).read().split("\n")
    out=[]; i=0; n=0
    while i < len(lines):
        l=lines[i]
        if l.startswith("<<<<<<< "):
            ours=[];base=[];theirs=[]
            i+=1
            while not lines[i].startswith("||||||| "):
                ours.append(lines[i]); i+=1
            i+=1
            while not lines[i].startswith("======="):
                base.append(lines[i]); i+=1
            i+=1
            while not lines[i].startswith(">>>>>>> "):
                theirs.append(lines[i]); i+=1
            i+=1
            sm=difflib.SequenceMatcher(None, base, theirs, autojunk=False)
            added=[]
            for tag,a0,a1,b0,b1 in sm.get_opcodes():
                if tag in ("insert","replace"):
                    added+=theirs[b0:b1]
            out+=ours+added
            n+=1
        else:
            out.append(l); i+=1
    open(path,"w").write("\n".join(out))
    return n
for p in sys.argv[1:]:
    print(p, resolve(p), "conflicts resolved")
