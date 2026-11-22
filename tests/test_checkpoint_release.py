"""Checkpoint inventory and reference-prediction navigation checks."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class CheckpointReleaseTests(unittest.TestCase):
    def test_checkpoints_and_native_backend_selection(self):
        cards=json.loads((ROOT/'MODEL_CARDS.json').read_text())
        self.assertEqual(len(cards['models']),11)
        for item in cards['models']:
            self.assertTrue(item['file'].endswith('.pt'))
            self.assertTrue(item['portable_weights'])
            self.assertEqual(len(item['native_type_map']),118)
            self.assertEqual(item['backend_selector'],'pt' if item['architecture']=='dpa4' else 'pt-expt')

    def test_reference_predictions_match_checkpoint_identities(self):
        cards=json.loads((ROOT/'MODEL_CARDS.json').read_text())
        identities={(x['system'],x['scientific_role']):x for x in cards['models']}
        refs=json.loads((ROOT/'FIRST_FRAME_REFERENCES.json').read_text())
        self.assertEqual(refs['schema'],2)
        self.assertEqual(len(refs['outputs']),6)
        for item in refs['outputs']:
            identity=identities[item['system'],item['scientific_role']]
            self.assertEqual(item['model_sha256'],identity['sha256'])
            self.assertEqual(item['model_file'],identity['file'])
            self.assertEqual(len(item['archived_prediction_forces_eV_A']),item['atom_count'])
            self.assertEqual(len(item['archived_prediction_virial_eV']),3)

    def test_recorded_native_environment(self):
        runtime=json.loads((ROOT/'NATIVE_ENVIRONMENT.json').read_text())
        self.assertEqual(runtime['deepmd_version'],'3.2.0+g3b2555c46')
        self.assertEqual(runtime['deepmd_revision'],'3b2555c46ecdbd2fcc8ddc029d09195a4e3c0f3a')


if __name__=='__main__':unittest.main()
