import unittest
import numpy as np
from evaluate_models import evaluate_frame

class Atoms:
    pbc=np.array([True]*3)
    positions=np.array([[0.,0.,0.],[1.,0.,0.]])
    class Cell: array=np.eye(3)*10
    cell=Cell()
    def get_chemical_symbols(self): return ['O','Al']

class Potential:
    def get_type_map(self): return ['H','Al','O']
    def eval(self,coords,cells,types):
        assert types==[2,1]
        assert coords.shape==(1,2,3) and cells.shape==(1,9)
        return np.array([[4.]]),np.zeros((1,2,3)),np.zeros((1,9))

class TestModelExample(unittest.TestCase):
    def test_native_species_indices_and_shapes(self):
        e,f,v=evaluate_frame(Potential(),Atoms())
        self.assertEqual(e,4.)
        self.assertEqual(f.shape,(2,3)); self.assertEqual(v.shape,(3,3))

if __name__=='__main__':unittest.main()
