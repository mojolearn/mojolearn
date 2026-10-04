"""CPU-only metadata/concurrency checks; run on authorized Linux boxes."""
import threading
import unittest
from identical_wave_runner import compile_batch
from identical_sgd_dependency_proof import reachable, prove

class PrepareTests(unittest.TestCase):
    def test_parallel_failures_and_exceptions_survive_success(self):
        barrier=threading.Barrier(3)
        def worker(name):
            barrier.wait(timeout=5)
            if name=='exception': raise RuntimeError('fixture failure')
            return {'id':name,'rc':9 if name=='failed' else 0}
        steps=[]
        self.assertEqual(compile_batch(['success','failed','exception'],worker,3,steps),1)
        self.assertEqual({x['id']:x['rc'] for x in steps},{'success':0,'failed':9,'exception':127})

    def test_serial_success(self):
        steps=[]
        self.assertEqual(compile_batch(['a','b'],lambda x:dict(id=x,rc=0),1,steps),0)
        self.assertEqual([x['id'] for x in steps],['a','b'])

    def test_graph_follows_transitive_cycle(self):
        graph={'edges':{'entry':['helper'],'helper':['device'],'device':['helper']}}
        self.assertEqual(reachable(graph,'entry'),{'entry','helper','device'})

    def test_changed_gpu_module_never_reuses(self):
        with self.assertRaisesRegex(RuntimeError,'cannot be reused'):
            prove(None,'x_linear',None,[])

if __name__=='__main__':unittest.main()
