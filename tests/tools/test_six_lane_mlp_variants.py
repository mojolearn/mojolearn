"""Metadata-only rejection tests; never imports/constructs a product estimator."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import six_lane_mlp_variants as m
from six_lane_register_full_mlp import facts_for
from six_lane_full_variants import validate_variant
from six_lane_ab import runtime_requirements


class MLPAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The one scope cell this test needs, kept from the retired six-lane matrix (deleted 2026-10-08).
        cls.scope=json.loads((Path(__file__).with_name('six_lane_mlp_scope_cell.json')).read_text())
        cls.matrix={'cells':[cls.scope]}

    def fact(self,row):
        cell=m.variant_cell(self.scope,row)
        with patch.object(m,'retained',return_value=row['sidecar_metadata']):
            fact=facts_for(cell,'test-freeze',Path('/retained/full'))
        return cell,fact

    def test_four_new_distinct_cells_preserve_original(self):
        original=copy.deepcopy(self.matrix['cells'])
        expanded=m.append_registered_cells(original)
        self.assertEqual(expanded[:len(original)],original)
        new=expanded[len(original):]
        self.assertEqual(len(new),4)
        self.assertEqual(len({c['key'] for c in new}),4)
        for row in m.contracts()['rows']:
            cell,fact=self.fact(row)
            with patch.object(m,'retained',return_value=row['sidecar_metadata']):validate_variant(fact,cell)

    def test_reject_settings_caps_hash_scope_and_marker_changes(self):
        row=m.contracts()['rows'][0];cell,fact=self.fact(row)
        mutations=[lambda f:f['workload']['estimator_settings_record']['params'].update(max_iter=1),
            lambda f:f['workload']['input_files'][0].update(sha256='0'*64),
            lambda f:f.update(intrinsic_caps=['smaller rows']),
            lambda f:f.update(changes_frozen_race=False),
            lambda f:f['registered_input_variant'].update(original_cell_key='wrong'),
            lambda f:f['registered_input_variant'].update(measurement_source_sha='wrong'),
            lambda f:f['workload'].update(inference='included_in_operation'),
            lambda f:f['workload'].update(output_paths=[])]
        for mutate in mutations:
            bad=copy.deepcopy(fact);mutate(bad)
            with self.subTest(mutate=mutate),patch.object(m,'retained',return_value=row['sidecar_metadata']):
                with self.assertRaises(ValueError):validate_variant(bad,cell)

    def test_reject_sidecar_and_vendor_changes(self):
        row=m.contracts()['rows'][0];cell,fact=self.fact(row)
        meta=copy.deepcopy(row['sidecar_metadata']);meta['full_dataset_coverage']=False
        with patch.object(m,'retained',return_value=meta):
            with self.assertRaises(ValueError):validate_variant(fact,cell)
        with patch.object(m,'retained',return_value=row['sidecar_metadata']):
            with self.assertRaises(ValueError):validate_variant(fact,dict(cell,vendor='amd'))
            bad=copy.deepcopy(fact);bad['artifact_provenance']={'A':[],'B':[]}
            with self.assertRaises(ValueError):validate_variant(bad,cell)
        wrong=dict(self.scope,vendor='amd')
        with self.assertRaises(ValueError):m.variant_cell(wrong,row)

    def test_actual_worker_job_requires_both_binding_closures(self):
        row=m.contracts()['rows'][0];cell,fact=self.fact(row)
        fact['vendor']='nvidia'
        fact['job']=dict(key=cell['key'],workload_id=cell['workload_id'],mode=cell['mode'],
                        master_selection=dict(id=cell['configuration']))
        with tempfile.TemporaryDirectory() as temp:
            items=[]
            for index,binding in enumerate(row['required_bindings']):
                path=Path(temp)/(str(index)+'.json')
                path.write_text(json.dumps(dict(binding=binding)))
                items.append(dict(receipt=str(path)))
            full={'A':items,'B':items}
            # A valid top-level deployment cannot fill a missing worker job closure.
            fact['artifact_provenance']=copy.deepcopy(full)
            with patch.object(m,'retained',return_value=row['sidecar_metadata']):
                for missing in (None,{}, {'A':items}, {'A':items[:1],'B':items},
                                {'A':items,'B':items[:1]}):
                    bad=copy.deepcopy(fact)
                    if missing is not None:bad['job']['artifact_provenance']=missing
                    with self.subTest(provenance=missing):
                        with self.assertRaises(ValueError):validate_variant(bad,cell)
                fact.pop('artifact_provenance')
                fact['job']['artifact_provenance']=full
                validate_variant(fact,cell)

    def test_unknown_variant_remains_rejected(self):
        with self.assertRaises(ValueError):validate_variant(dict(changes_frozen_race=True,registered_input_variant=dict(variant='anything')))

    def test_only_reviewed_full_recipe_resolves_unrelated_runtime_controls(self):
        cfgs={c['id']:c for c in self.matrix['configurations']};cfg=cfgs['I.X.complete-proposed']
        for row in m.contracts()['rows']:
            self.assertEqual(runtime_requirements(cfg,row['variant_workload_id'],cfgs),{})
        self.assertTrue(runtime_requirements(cfg,'expanded:unknown@input=mlp-full-v1',cfgs))


if __name__=='__main__':unittest.main()
