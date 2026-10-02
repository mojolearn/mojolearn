import sys,re
box,f=sys.argv[1],sys.argv[2]
sec=None;fam=None;seen=set();ours=None;opp=[]
def flush():
    if sec and ours and opp:
        b=min(opp,key=lambda x:x[1]); print("%s\t%s\t%.1f\t%s\t%.1f\t%.2f"%(box,sec,ours,b[0],b[1],ours/b[1]))
for line in open(f):
    if line.startswith("## "): fam=line[3:].strip()
    if line.startswith("### ") and fam=="Trees":
        flush(); sec=line[4:].split("(")[0].strip(); ours=None; opp=[]; done=False; continue
    if sec and line.startswith("Inference"): flush(); sec=None
    if sec and line.startswith("| ") and not line.startswith("| arm") :
        c=[x.strip() for x in line.split("|")]
        try: ms=float(c[5])
        except: continue
        if c[1]=="mojolearn IDENTICAL": ours=ms
        elif c[4]=="opponent": opp.append((c[1],ms))
flush()
