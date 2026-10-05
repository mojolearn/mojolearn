"""Candidate build planning tests: no compiler, model, or GPU execution."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import identical_wave_native_build as b


class CandidateBuildTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        subprocess.run(['git','init','-q',str(self.repo)],check=True)
        (self.repo/'tools').mkdir()
        source = 'from std.sys.compile import is_defined\ncomptime X = is_defined["MOJOLEARN_IDN_TEST"]()\n'
        source += '# MOJOLEARN_IDN_COMMENT_ONLY is not a compiled switch\n'
        (self.repo/'switch.mojo').write_text(source)
        doc={'recipes':[{'id':'test/1','baseline_defines':[],
                         'candidate_defines':['MOJOLEARN_IDN_TEST=1'],
                         'recommended_builders':['base','base_host']}]}
        (self.repo/'tools/identical_candidate_recipes.json').write_text(json.dumps(doc))
        subprocess.run(['git','-C',str(self.repo),'add','.'],check=True)
        subprocess.run(['git','-C',str(self.repo),'-c','user.name=Fixture',
                        '-c','user.email=fixture@example.invalid','commit','-qm','fixture'],check=True)
        self.sha=subprocess.check_output(['git','-C',str(self.repo),'rev-parse','HEAD'],text=True).strip()

    def test_tokens_are_normalized_without_shell_syntax(self):
        self.assertEqual(b.explicit_defines(['MOJOLEARN_IDN_TEST','MOJOLEARN_IDN_ROWS=256']),
                         ['MOJOLEARN_IDN_TEST=1','MOJOLEARN_IDN_ROWS=256'])
        for bad in ['MOJOLEARN_IDN_X=0','MOJOLEARN_IDN_X=-1','MOJOLEARN_IDN_X=1;id',
                    'MOJOLEARN_IDN_X=$(id)','MOJOLEARN_IDN_X=1 -D OTHER','OTHER=1']:
            with self.subTest(bad=bad),self.assertRaises(ValueError):b.explicit_defines([bad])

    def test_reserved_and_duplicate_flags_refused(self):
        for bad in ['MOJOLEARN_IDN_ALL_OFF','MOJOLEARN_COLUMN_AMD','MOJOLEARN_NUMERIC_IDENTICAL']:
            with self.assertRaises(ValueError):b.explicit_defines([bad])
        with self.assertRaises(ValueError):b.explicit_defines(['MOJOLEARN_IDN_TEST','MOJOLEARN_IDN_TEST=2'])

    def test_unknown_or_comment_only_flags_refused(self):
        locations=b.define_source_locations(self.repo,self.sha,['MOJOLEARN_IDN_TEST=1'])
        self.assertEqual(locations['MOJOLEARN_IDN_TEST'][0]['path'],'switch.mojo')
        for name in ['MOJOLEARN_IDN_MISSING=1','MOJOLEARN_IDN_COMMENT_ONLY=1']:
            with self.assertRaises(ValueError):b.define_source_locations(self.repo,self.sha,[name])

    def test_recipe_comes_from_commit_not_dirty_checkout(self):
        path=self.repo/'tools/identical_candidate_recipes.json'
        path.write_text('{"recipes":[]}')
        row,defs,sha=b.candidate_recipe(self.repo,self.sha,'test/1','candidate')
        self.assertEqual(defs,['MOJOLEARN_IDN_TEST=1'])
        self.assertEqual(len(sha),64)
        self.assertEqual(b.candidate_recipe(self.repo,self.sha,'test/1','baseline')[1],[])
        with self.assertRaises(ValueError):b.candidate_recipe(self.repo,self.sha,'missing','candidate')


if __name__=='__main__':unittest.main()
