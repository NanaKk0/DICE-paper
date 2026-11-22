"""CPU checks for current evidence identities, formulas and navigation."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reproduce import Archive,assert_close,summary
from paper_figures import rebuild,PaperPlotter
import matplotlib.pyplot as plt


class CurrentReleaseTests(unittest.TestCase):
    def setUp(self):self.a=Archive(ROOT/'source-data')

    def test_archive_path_boundary(self):
        with self.assertRaises(ValueError):self.a.path('../outside')

    def test_object_inventory(self):
        self.assertEqual(len(self.a.registry['figures']),20)
        self.assertEqual(len(self.a.registry['tables']),5)
        for figure in self.a.registry['figures']:
            for path in [figure['artwork'],*figure['sources']]:self.assertTrue(self.a.path(path).is_file(),path)

    def test_current_panel_and_object_maps(self):
        with (ROOT/'MAIN_PANEL_MAP.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual([r['panel'] for r in rows if r['figure']=='Fig. 2'],list('ab'))
        self.assertEqual([r['panel'] for r in rows if r['figure']=='Fig. 3'],list('abcde'))
        for row in rows:
            for path in row['input_files'].split(' ; '):self.assertTrue((ROOT/path).is_file(),path)
        with (ROOT/'FIGURE_TABLE_MAP.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),26)

    def test_70_frame_design_and_heldout_counts(self):
        d=self.a.csv('data/controlled/metrics_per_repeat.csv')
        held=d[d.scope=='excluded_training_frames']
        self.assertEqual(set(held.loc[held.pool=='liquid423','frames']),{363})
        self.assertEqual(set(held.loc[held.pool=='gamma506','frames']),{446})
        self.assertEqual(set(held.route),{'T0_70','Hybrid_70','TR_70','DFT_20','DFT_70'})

    def test_paired_tail_formula(self):
        d=self.a.csv('data/controlled/Fig2a_paired_tails.csv')
        assert_close(d.tail_mean_eV_A-d.baseline_tail_eV_A,d.difference_eV_A,'Tail differences')

    def test_relative_p99_mean_uses_per_repeat_ratios(self):
        d=self.a.csv('data/routing/Fig3g_relative_P99_paired_source.csv')
        self.assertEqual(len(d),50)
        assert_close(100*(d.teacher_P99_eV_A/d.DFT_P99_eV_A-1),d.relative_P99_difference_percent,'Relative P99')
        self.assertEqual(set(d.teacher_label_source),{'T0','T1'})

    def test_accepted_fraction_denominator(self):
        d=self.a.csv('data/si_comparisons/routing_acceptance_per_repeat.csv')
        assert_close(100*d.high_error_accepted/d.accepted,d.high_error_fraction_percent,'Accepted fraction')

    def test_mixing_all_points_and_formula(self):
        for element,n in [('Al',12480),('O',18720)]:
            d=self.a.csv(f'data/crossphase/TSNE_{element}.csv')
            self.assertEqual(len(d),n)
            u=d.teacher_density/(d.teacher_density+d.test_density)
            assert_close(4*u*(1-u),d.local_mixing,'Neighbourhood mixing')

    def test_unknown_figure_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):rebuild(self.a,Path(d),['main:99'])

    def test_cost_bar_labels_follow_numeric_system_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=PaperPlotter(self.a,Path(d));saved=[]
            p.save=lambda fig,*args,**kw:saved.append(fig)
            p.main7();fig=saved[0];ax=fig.axes[1]
            self.assertEqual([t.get_text() for t in ax.get_xticklabels()],[r'Al$_2$O$_3$','YSZ','MIL-120'])
            heights=[patch.get_height() for patch in ax.patches]
            assert_close(heights[1::2],[304.1386991056702,202.12822089742917,229.82436676493182],'Labelled teacher bars')
            plt.close(fig)

    def test_summary_reconstructs_cost_and_key_claims(self):
        with tempfile.TemporaryDirectory() as d:
            result=summary(self.a,Path(d))
        self.assertEqual(result['cost_reanalysis']['task_work_records'],75038)
        self.assertAlmostEqual(result['teacher_reduction_percent'],31.6743495658,places=8)
        costs={x['system']:x['teacher_estimated_h'] for x in result['cost_reanalysis']['systems']}
        self.assertAlmostEqual(costs['Al2O3/AL3'],304.1386991057,places=8)
        self.assertAlmostEqual(costs['YSZ/AL2'],202.1282208974,places=8)
        self.assertAlmostEqual(costs['MOF-MIL-120'],229.8243667649,places=8)


if __name__=='__main__':unittest.main()
