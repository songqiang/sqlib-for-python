import unittest
import sys
import pandas as pd
import numpy as np

sys.path.append('../')
from pandasutil import *

class Test(unittest.TestCase):

    def test_pdcompare(self):
        df1 = pd.DataFrame([(1, 2, 's1', 'k1'), (3, 4, 's1', 'k2')],
                           columns = ['n1', 'n2', 'c', 'k'])

        df2 = pd.DataFrame([(1, 2, 'x1', 'k1'), (2, 1, 's1', 'k2')],
                           columns = ['n2', 'n1', 'c', 'k'])

        pdcompare(df1, df2)
        pdcompare(df1, df2, ['k'])

if __name__ == '__main__':
    unittest.main()
