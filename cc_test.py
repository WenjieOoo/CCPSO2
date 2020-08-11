import random
import numpy as np
from fitnessfunction import function1, function2
from CCPSO2 import CCPSO2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from fitnessfunction import function3
from tqdm import tqdm

# test
print("GO...")
ccpso2 = CCPSO2(function2, 20, 1000)
result = ccpso2.evolve()

print()
