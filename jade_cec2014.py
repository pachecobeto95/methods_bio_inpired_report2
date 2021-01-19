import numpy as np
import random
import sys, math
from differential_evolution import JADE
import pandas as pd
from utils import write_parameters_de, write_performance, write_evolution
#from optproblems import cec2005
from pathlib import Path
#from opfunu.cec.cec2014.function import F1, F2, F4, F6, F7, F9, F14
#Funções: (1,2,4,6,7,9,14).
import sys
sys.path.append("/home/pacheco/cec2014/python")
#import cec2014
from cec2014 import cec14

class FuncObj():
	def __init__(self, i):
		self.i = i

	def func(self, arr):
		return cec14(arr, self.i)


def experiment(f, n_runs, pop_size, n_epochs, dim, bounds, parameters_dict, optimum, max_obj=False):

	error_list, fes_list, error_evolution_list  = [], [], []
	success_count = 0
	for n in range(n_runs):
		print("Run: %s"%(n))
		code = JADE(f, pop_size, n_epochs, dim, bounds, parameters_dict, optimum, max_obj=False)
		error, success, error_evolution, fes = code.run(show_info=True)
		error_list.append(error)
		fes_list.append(fes)
		error_evolution_list.append(error_evolution)
		success_count+=success

	success_rate = success_count/n_runs
	return error_list,fes_list, error_evolution_list, success_rate 



bounds = [-100, 100]
n_epochs = 10000
pop_size = 70
dim = 30
cross_rate = .9
impact_factor = .5
max_obj = False
optimum_list = np.array([100, 200, 400, 600, 700, 900, 1400])
f_id_list = optimum_list/100
function_list = [FuncObj(f_id).func for f_id in f_id_list]

n_runs = 25
crossover_method = "binomial"
mutation_method = "JADE"
initialization_method = "random"
n_diff = 1
archive_size = pop_size
parameters_dict = {"cross_method": crossover_method, "cross_rate": cross_rate,
"mutation_method": mutation_method, "impact_factor":impact_factor, "nr_diff": n_diff,
"initialization_method": initialization_method, "jumping_rate": 0.3, "archive_size": archive_size}

savePathParameters = Path("new_new_reports_final/jade_parameters_dim_%s.csv"%(dim))
savePathReporter = Path("new_new_reports_final/jade_statistics_dim_%s.csv"%(dim))

for i, f, optimum in zip(f_id_list, function_list, optimum_list):
	savePathEvolution = Path("new_new_reports_final/f%s_%s_evolution_error_dim_%s.csv"%(i, mutation_method, dim))

	error_list,fes_list, error_evolution_list, success_rate = experiment(f, n_runs, pop_size, n_epochs, dim, 
		bounds, parameters_dict, optimum, max_obj=False)


	write_performance("F%s"%(i), dim, error_list, fes_list, fes_list, pop_size, success_rate, savePathReporter, max_obj=False)
	write_parameters_de("F%s"%(i), n_epochs, dim, pop_size, parameters_dict, savePathParameters, max_obj=False)
	write_evolution(dim, error_evolution_list, n_runs, savePathEvolution)

