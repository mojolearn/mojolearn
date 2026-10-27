import copy
import json
import pytest
import main_board_ingest as M


def observation():
    cell = {'arm': 'ours', 'median_ms': 1.25, 'measurement': {'source': {'sha256': 'source'}, 'workers': [{'params': {'lr': 0.001}}]}}
    return {'sha': 'a' * 40, 'measured_cell': copy.deepcopy(cell),
            'paired_race': {'cells': [cell, {'arm': 'opponent', 'median_ms': 2.0}], 'audit': {'verdict': 'MATCHED'}}}


def test_pair_reference_round_trips_all_evidence_and_rejects_changes():
    original = observation()
    packed = M.compact_observation(original)
    assert 'measured_cell' not in packed
    assert M.restore_observation(packed) == original
    assert original == observation()
    packed = copy.deepcopy(packed)
    packed['paired_race']['cells'][0]['measurement']['workers'][0]['params']['lr'] = 0.002
    with pytest.raises(ValueError, match='bytes changed'):
        M.restore_observation(packed)


def test_different_cells_and_numeric_types_are_not_deduplicated():
    original = observation()
    original['measured_cell']['median_ms'] = 1
    original['paired_race']['cells'][0]['median_ms'] = 1.0
    assert M.compact_observation(original) == original


def test_board_compaction_changes_only_duplicate_observation_storage(tmp_path):
    import race_fairness as F
    original = observation()
    board = {'updated': 'unchanged', 'races': {'r': {'audit': {'verdict': 'MATCHED'},
             'cells': [{'arm': 'ours', 'mode': 'identical', 'library': 'mojolearn', 'median_ms': 1.25, 'main_board': {'observation': original}}]}}}
    path = tmp_path / 'board.json'; path.write_text(json.dumps(board))
    original_fingerprint = F.pairing(board['races']['r'], 'identical', 'fit')
    M.compact_existing_board(path)
    compact = json.loads(path.read_text())
    assert F.pairing(compact['races']['r'], 'identical', 'fit') == original_fingerprint
    mb = compact['races']['r']['cells'][0]['main_board']
    mb['observation'] = M.restore_observation(mb['observation'])
    assert compact == board
