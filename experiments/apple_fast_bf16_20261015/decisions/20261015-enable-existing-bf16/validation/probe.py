import sys,json,hashlib,pathlib,numpy as np
from mojolearn import _linalg_impl,linalg,linalg_fast
binding=_linalg_impl._load();variant=sys.argv[1];base=pathlib.Path(sys.argv[2]);out=pathlib.Path(sys.argv[3]);expected=sys.argv[4]
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
record={'variant':variant,'mode':int(binding.linalg_numeric_mode()),'binary':str(pathlib.Path(binding.__file__).resolve()),'binary_sha256':sha(binding.__file__),'fast_bf16_available':bool(binding.fast_bf16_available())}
assert record['binary_sha256']==expected
assert record['mode']==(1 if variant=='identical' else 0)
assert record['fast_bf16_available']==(variant=='fast-default')
if variant=='fast-off':
 try:linalg_fast.matmul_bf16(np.ones((2,2),np.float32),np.ones((2,2),np.uint16))
 except Exception as e:record['disabled_refusal']=str(e)
 else:raise AssertionError('disabled BF16 path accepted a call')
else:
 input_path=base/'input-4096-4096-4096.npz'
 with np.load(input_path,allow_pickle=False) as data:a,b=data['a'],data['b']
 result=np.asarray(linalg.matmul_bf16(a,b));ref=base/('candidate-fast' if variant=='fast-default' else 'released-identical')/'output-4096-4096-4096-0.npy';reference=np.load(ref,mmap_mode='r')
 assert result.shape==reference.shape and result.dtype==reference.dtype
 record.update(input_sha256=sha(input_path),reference_sha256=sha(ref),shape=list(result.shape),bitwise_equal=bool(np.array_equal(result.view(np.uint32),reference.view(np.uint32))),differing_elements=int(np.count_nonzero(result.view(np.uint32)!=reference.view(np.uint32))))
 np.save(out/(variant+'-output.npy'),result)
 record['output_sha256']=sha(out/(variant+'-output.npy'))
 assert record['bitwise_equal'],record
record['status']='passed';(out/(variant+'-probe.json')).write_text(json.dumps(record,indent=2));print(json.dumps(record))
