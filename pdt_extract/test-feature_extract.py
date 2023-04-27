import unittest
from feature_extract import FeatureExtract
import imageio
import numpy as np
from scipy import misc


class MyTestCase(unittest.TestCase):
    def test_something(self):
        indices = np.where(profile == 255)
        x = np.flip(indices[1])
        y = np.flip(indices[0])
        test_extractor = FeatureExtract(x, y)


if __name__ == '__main__':
    unittest.main()
