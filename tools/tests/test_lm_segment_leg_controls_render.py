"""tools/lm_segment_leg.py controls: the default render is the NVIDIA two-step
controls body as before; `--arm amd --devices 0,1 --steps 3 --controls ""`
renders a rehearsal body for an AMD box that holds the checkpoint's line and
the three after it. No URL is minted (presigning is mocked)."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.argv = [sys.argv[0]]
TOOLS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('lm_segment_leg', TOOLS / 'lm_segment_leg.py')
leg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(leg)


class ControlsRender(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.recipe = d / 'recipe.json'
        self.recipe.write_text('{"steps": 5000}\n')
        self.chain = d / 'chain.jsonl'
        self.chain.write_text(''.join(json.dumps(dict(step=s, state_sha256='%064x' % s)) + '\n' for s in range(95, 110)))
        self.out = d / 'body.sh'
        fake = types.ModuleType('lm_segment')
        fake.load_recipe = lambda path: {}
        self.patches = [mock.patch.dict(sys.modules, {'lm_segment': fake}),
                        mock.patch.object(leg, 'presign_get', side_effect=lambda key, s: 'https://get.example/' + key),
                        mock.patch.object(leg, 'tokens_urls', return_value={'tokens.i32.part00': 'https://get.example/t0'})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def render(self, *extra):
        with mock.patch.object(leg.subprocess, 'run'):
            leg.main(['controls', '--recipe', str(self.recipe), '--recipe-key', 'runs/t3/x/recipe.json',
                      '--from-key', 'runs/t3/x/A/1/ckpt_00000100.blm', '--tokens-key', 'tok',
                      '--expect-chain', str(self.chain), '--wheel', '0.8.17', '--out', str(self.out), *extra])
        return self.out.read_text()

    def lines(self, body):
        block = body.split("<<'EXPECT_CHAIN'\n", 1)[1].split('\nEXPECT_CHAIN', 1)[0]
        return [json.loads(line)['step'] for line in block.splitlines()]

    def test_default_is_the_nvidia_two_step_controls(self):
        body = self.render()
        self.assertIn('ARM="nvidia"', body)
        self.assertIn('DEVICES="0"', body)
        self.assertIn('STEPS="2"', body)
        self.assertEqual(self.lines(body), [100, 101, 102])
        self.assertIn('CONTROLS="none:none zero-moments:zero-moments', body)
        self.assertIn('--steps "$STEPS" --devices "$DEVICES"', body)
        for ph in ('@ARM@', '@DEVICES@', '@STEPS@', '@FROM_NAME@', '@CONTROLS@', '@WHEEL@', '@EXPECT_LINES@'):
            self.assertNotIn(ph, body)

    def test_amd_rehearsal(self):
        body = self.render('--arm', 'amd', '--devices', '0,1', '--steps', '3', '--controls', '')
        self.assertIn('ARM="amd"', body)
        self.assertIn('DEVICES="0,1"', body)
        self.assertIn('STEPS="3"', body)
        self.assertIn('CONTROLS=""', body)
        self.assertEqual(self.lines(body), [100, 101, 102, 103])
        self.assertIn('rocm-smi --showproductname --showbus --showuniqueid', body)
        for ph in ('@ARM@', '@DEVICES@', '@STEPS@', '@FROM_NAME@', '@CONTROLS@', '@WHEEL@', '@EXPECT_LINES@'):
            self.assertNotIn(ph, body)

    def test_a_chain_without_the_held_steps_is_refused(self):
        with self.assertRaises(SystemExit):
            self.render('--steps', '20')


if __name__ == '__main__':
    unittest.main()
