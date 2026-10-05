"""Synthetic gate metadata tests: no Mojo execution, GPU, timing or external data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from identical_wave_compare import classify_receipts, compare_receipts, sha256, validate_partial_cases, validate_proof
from identical_wave_runner import ProgressSteps, binary_inventory, verify_products, write


class IdentityProofTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.dumps({'cases': [{'lane': 'linear', 'dataset': 'medium'},
                                         {'lane': 'neural', 'dataset': 'small', 'driver': 'neural', 'fixture': 'fixed.npz'}]}).encode()
        self.receipts = {}
        for vendor, backend in [('nvidia', 'cuda'), ('amd', 'hip')]:
            receipt = {'phase': 'identity', 'status': 'PASS', 'identity': {
                'vendor': vendor, 'gpu_arch': {'nvidia': 'sm_120', 'amd': 'gfx942'}[vendor],
                'sha': 'a'*40, 'plan_sha256': sha256(self.plan),
                'data_manifest_sha256': 'b'*64, 'neural_fixture_sha256': {'fixed.npz': 'c'*64}}, 'arms': {}}
            for arm, digest in [('on', 'd'*64), ('off', 'e'*64)]:
                steps = []
                for tag in ['linear--medium', 'neural--small']:
                    steps += [{'id': tag+'--'+v, 'rc': 0, 'status': 'PASS'} for v in [backend, 'cpu']]
                    steps.append({'id': tag+'--bits', 'status': 'PASS', 'digests': {backend: digest, 'cpu': digest}})
                receipt['arms'][arm] = steps
            self.receipts[vendor] = receipt

    def raw(self, vendor):
        return json.dumps(self.receipts[vendor]).encode()

    def compare(self):
        return compare_receipts(self.raw('nvidia'), self.raw('amd'), self.plan)

    def test_valid_proof_and_different_bits_between_arms(self):
        proof = self.compare()
        for vendor in ['nvidia', 'amd']:
            validate_proof(proof, self.raw(vendor), vendor, self.plan)
        self.assertEqual(proof['receipt_sha256']['amd'], sha256(self.raw('amd')))

    def test_reject_source_data_plan_or_neural_mismatch(self):
        for key, value in [('sha', 'f'*40), ('data_manifest_sha256', 'f'*64),
                           ('plan_sha256', 'f'*64), ('neural_fixture_sha256', {'fixed.npz': 'f'*64})]:
            with self.subTest(key=key):
                original = copy.deepcopy(self.receipts['amd'])
                self.receipts['amd']['identity'][key] = value
                with self.assertRaises(ValueError): self.compare()
                self.receipts['amd'] = original

    def test_reject_cross_vendor_digest_disagreement_even_when_local_agrees(self):
        self.receipts['amd']['arms']['on'][2]['digests'] = {'hip': 'f'*64, 'cpu': 'f'*64}
        with self.assertRaisesRegex(ValueError, 'cross-vendor digest'): self.compare()

    def test_reject_missing_cases_on_both_vendors(self):
        for receipt in self.receipts.values(): receipt['arms']['off'] = receipt['arms']['off'][:3]
        with self.assertRaisesRegex(ValueError, 'missing'): self.compare()

    def test_reject_missing_arm_or_digest_or_failed_receipt(self):
        original = copy.deepcopy(self.receipts)
        mutations = [lambda r: r['arms'].pop('off'),
                     lambda r: r['arms']['on'][2]['digests'].pop('cpu'),
                     lambda r: r.update(status='INCOMPLETE')]
        for mutate in mutations:
            self.receipts = copy.deepcopy(original); mutate(self.receipts['amd'])
            with self.assertRaises(ValueError): self.compare()

    def test_reject_receipt_bytes_changed_after_comparison(self):
        proof = self.compare(); self.receipts['nvidia']['note'] = 'changed receipt'
        with self.assertRaisesRegex(ValueError, 'receipt hash'):
            validate_proof(proof, self.raw('nvidia'), 'nvidia', self.plan)

    def test_arch_is_vendor_specific_and_bound_to_receipt(self):
        proof = self.compare()
        self.receipts['nvidia']['identity']['gpu_arch'] = 'sm_89'
        with self.assertRaisesRegex(ValueError, 'receipt hash'):
            validate_proof(proof, self.raw('nvidia'), 'nvidia', self.plan)
        self.receipts['nvidia']['identity']['gpu_arch'] = 'gfx942'
        with self.assertRaisesRegex(ValueError, 'architecture'): self.compare()
        self.receipts['nvidia']['identity'].pop('gpu_arch')
        with self.assertRaisesRegex(ValueError, 'fields'): self.compare()

    def test_reject_stale_or_incomplete_proof(self):
        for mutation in [lambda p: p['identity_core'].update(sha='f'*40),
                         lambda p: p['arms']['off'].pop('linear--medium'),
                         lambda p: p['arms']['on']['linear--medium']['amd'].update(hip='f'*64)]:
            proof = self.compare(); mutation(proof)
            with self.assertRaises(ValueError): validate_proof(proof, self.raw('nvidia'), 'nvidia', self.plan)


class BinaryInventoryTests(unittest.TestCase):
    def test_reject_added_removed_and_modified_libraries(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp); directory = root/'python/mojolearn'; directory.mkdir(parents=True)
            original = directory/'original.so'; original.write_bytes(b'original')
            recorded = binary_inventory(root); verify_products(root, recorded)
            extra = directory/'extra.so'; extra.write_bytes(b'extra')
            with self.assertRaises(ValueError): verify_products(root, recorded)
            extra.unlink(); original.unlink()
            with self.assertRaises(ValueError): verify_products(root, recorded)
            original.write_bytes(b'replaced')
            with self.assertRaises(ValueError): verify_products(root, recorded)

    def test_step_progress_is_durable(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            receipt = Path(tmp)/'progress.json'; report = {'status': 'INCOMPLETE', 'arms': {}}
            steps = ProgressSteps(lambda: write(receipt, report)); report['arms']['on'] = steps
            steps.append({'id': 'finished-step', 'rc': 0})
            saved = json.loads(receipt.read_text())
            self.assertEqual(saved['arms']['on'][0]['id'], 'finished-step')
            self.assertEqual(saved['status'], 'INCOMPLETE')


class PartialProofTests(IdentityProofTests):
    def test_partial_classes_and_subset_gate(self):
        amd_on = self.receipts['amd']['arms']['on']
        for step in amd_on:  # host column missing for one case on AMD ON
            if step['id'] == 'neural--small--cpu':
                step.update(rc=1, status='FAIL')
            if step['id'] == 'neural--small--bits':
                step.update(status='FAIL', digests={'hip': 'd'*64})
        proof = classify_receipts(self.raw('nvidia'), self.raw('amd'), self.plan)
        self.assertEqual(proof['status'], 'PARTIAL')
        self.assertEqual(proof['classes']['on'], {'PROVEN': ['linear--medium'], 'GPU_ONLY_MATCH': ['neural--small']})
        validate_partial_cases(proof, self.raw('amd'), 'amd', self.plan, ['linear--medium'])
        with self.assertRaises(ValueError):
            validate_partial_cases(proof, self.raw('amd'), 'amd', self.plan, ['neural--small'])

    def test_partial_differ(self):
        for step in self.receipts['amd']['arms']['off']:
            if step['id'] == 'linear--medium--bits':
                step['digests'] = {'hip': 'f'*64, 'cpu': 'f'*64}
        proof = classify_receipts(self.raw('nvidia'), self.raw('amd'), self.plan)
        self.assertEqual(proof['classes']['off'].get('DIFFER'), ['linear--medium'])
        with self.assertRaises(ValueError):
            validate_partial_cases(proof, self.raw('nvidia'), 'nvidia', self.plan, ['linear--medium'])

if __name__ == '__main__': unittest.main()
