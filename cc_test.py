import random
import numpy as np
from fitnessfunction import function1, function2
from CCPSO2 import CCPSO2
from cec2013lsgo.cec2013 import Benchmark
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from fitnessfunction import function3
from tqdm import tqdm
import pandas as pd


def main(fun_id):
    bench = Benchmark()

    i_dimension_size = 1000                         # 1000
    i_population_size = 30                          # 30
    max_number_of_fitness_evaluations = 5000000     # 5,000,000
    ifun = bench.get_function(fun_id)
    i_x_lower = bench.get_info(fun_id)['lower']
    i_x_upper = bench.get_info(fun_id)['upper']

    print("\nFunction Info: " + str(bench.get_info(fun_id)))
    print('<(￣︶￣)↗[GO!]')

    ccpso2 = CCPSO2(fun=ifun, dimension_size=i_dimension_size, population_size=i_population_size,
                    max_number_of_fitness_evaluations=max_number_of_fitness_evaluations, x_lower=i_x_lower,
                    x_upper=i_x_upper)
    gbest_history, X, gbest = ccpso2.evolve()

    pd.DataFrame(data=gbest_history).to_csv('results/CCPSO2_gbest_history_fun' + str(fun_id) + '.csv')
    pd.DataFrame(data=X).to_csv('results/X_result_fun' + str(fun_id) + '.csv')
    print(str(gbest))


if __name__ == '__main__':
    main(fun_id=1)
