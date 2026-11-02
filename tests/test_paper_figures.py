"""Regression tests for paper-panel semantics; no native model operations."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reproduce import Archive
from paper_figures import PaperPlotter, configure, rebuild
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


class PaperFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def test_unknown_figure_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):rebuild(Archive(ROOT/'source-data'),Path(tmp),['main:99'])

    def test_native_main3_restores_rounds_both_supports_and_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=PaperPlotter(Archive(ROOT/'source-data'),Path(tmp));captured=[]
            def capture(fig,stem,panels,selections=None):
                captured.append((fig,panels,selections))
            p.save=capture;p.main3();fig,panels,notes=captured[0]
            self.assertEqual(panels,list('abcdefg'))
            self.assertEqual(len(fig.axes),9)
            self.assertEqual(list(fig.axes[1].lines[0].get_xdata()),list(range(28)))
            self.assertEqual(notes['highlight'],[3,10])
            self.assertEqual(len(fig.axes[3].patches),4) # two filled/unfilled histograms
            self.assertEqual(len(fig.axes[5].patches),4)
            for ax,cn in zip(fig.axes[-3:],[4,5,6]):
                self.assertEqual(sum(len(x.get_offsets()) for x in ax.collections),2*cn+1)
            plt.close(fig)

    def test_main_phonons_exclude_only_disconnected_secondary_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=PaperPlotter(Archive(ROOT/'source-data'),Path(tmp));fig,ax=plt.subplots()
            p.dispersion(ax,matched=True)
            bands=[line for line in ax.lines if len(line.get_xdata())>2]
            self.assertEqual(len(bands),90)
            self.assertEqual(len(ax.get_xticks()),8)
            self.assertLess(min(min(line.get_ydata()) for line in bands),0)
            plt.close(fig)

    def test_route_aliases_and_read_only_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=PaperPlotter(Archive(ROOT/'source-data'),Path(tmp))
            self.assertEqual(p.route_style('A-DFTref')[0],'DFT-60')
            with self.assertRaises(ValueError):p.route_style('unverified historical route')
            code=(ROOT/'paper_figures.py').read_text()
            self.assertNotIn('torch.load',code)
            self.assertNotIn('subprocess.run',code)
            self.assertNotIn('np.random',code)

    def test_mil_eos_uses_single_points_on_shared_dft_geometries(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=PaperPlotter(Archive(ROOT/'source-data'),Path(tmp));fig,ax=plt.subplots()
            p.mil_eos_panel(ax,'A')
            d=p.csv('data/prediction_and_mil/all_EOS.csv')
            for line,role in zip(ax.lines,['DFT','Direct','DICE']):
                g=d[(d.phase=='A')&(d.route==('DICE' if role=='DFT' else role))].sort_values('volume_A3')
                expected=g.dft_relative_meV if role=='DFT' else g.sp_relative_meV
                np.testing.assert_allclose(line.get_ydata(),expected)
            plt.close(fig)

    def test_main_comparison_keeps_repeats_iqr_and_scattering_spread(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=PaperPlotter(Archive(ROOT/'source-data'),Path(tmp));saved=[]
            p.save=lambda fig,*args,**kwargs:saved.append(fig)
            p.main2();fig=saved.pop()
            repeat_points=[c for c in fig.axes[0].collections if isinstance(c,matplotlib.collections.PathCollection)]
            self.assertEqual(sum(len(c.get_offsets()) for c in repeat_points),22)
            plt.close(fig)

    def test_supplement_pairing_and_common_tail_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=PaperPlotter(Archive(ROOT/'source-data'),Path(tmp));saved=[]
            p.save=lambda fig,*args,**kwargs:saved.append(fig)
            p.supplement(3);fig=saved.pop()
            self.assertEqual(fig.axes[0].get_ylim(),fig.axes[1].get_ylim())
            self.assertEqual(fig.axes[2].get_ylim(),fig.axes[3].get_ylim())
            self.assertLess(fig.axes[2].get_ylim()[0],0)
            plt.close(fig)
            p.supplement(4);fig=saved.pop()
            for ax in fig.axes:
                paired=[line for line in ax.lines if len(line.get_xdata())==2]
                self.assertEqual(len(paired),5)
            plt.close(fig)
            p.main4();fig=saved.pop()
            for ax in fig.axes[:3]:
                bars=[c for c in ax.collections if isinstance(c,matplotlib.collections.LineCollection)]
                self.assertEqual(len(bars),2)
            plt.close(fig)
            p.main5();fig=saved.pop()
            for ax in fig.axes[1:3]:
                self.assertEqual(len(ax.collections),1)
            plt.close(fig)


if __name__=='__main__':unittest.main()
