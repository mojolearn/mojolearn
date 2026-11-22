import pathlib,sys
import mojolearn
sys.path.insert(0,str(pathlib.Path(__file__).parent/'tools'))
import par_witness_arm as w
w.HARNESS=pathlib.Path(mojolearn.__file__).parent/'_identity_break.py'
raise SystemExit(w.main())
