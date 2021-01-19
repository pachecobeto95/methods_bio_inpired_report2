import numpy as np
import random
import sys, math
from differential_evolution import DE
import pandas as pd
from utils import write_parameters_de, write_performance, write_evolution
#from optproblems import cec2005
from pathlib import Path
#from opfunu.cec.cec2014.function import F1, F2


import sys
sys.path.append("/home/pacheco/cec2014/python")
#import cec2014
from cec2014 import cec14

class FuncObj():
	def __init__(self, i):
		self.i = i

	def func(self, arr):
		return cec14(arr, self.i)


#def Func_obj(X):
#	z = []
#	for ind in X:
		#z.append(function1(ind))
#	return z


bounds = [-100, 100]
n_epochs = 10000
pop_size = 100
dim = 10
cross_rate = 0.7
impact_factor = 0.9
mutation_method = "variant"
crossover_method = "binomial"
max_obj = False
#optimum = 100.0
n_runs = 25
parameters_dict = {"cross_method": crossover_method, "cross_rate": cross_rate,
"mutation_method": mutation_method, "base": "best", "impact_factor":impact_factor, "nr_diff": 2}
savePathParameters = Path("new_new_reports_final/classic_de_parameters_dim_%s.csv"%(dim))
savePathReporter = Path("new_new_reports_final/classic_de_statistics_dim_%s.csv"%(dim))

#function1 = cec2005.F5(dim)
#function1 = F1
#optimum_list = np.array([100, 200, 400, 600, 700, 900, 1400])
optimum_list = np.array([700])
f_id_list = optimum_list/100
#function_list = [FuncObj(f_id).func for f_id in f_id_list]
function_list = [FuncObj(7).func for f_id in f_id_list]

for i, f, optimum in zip(f_id_list, function_list, optimum_list):
	savePathEvolution = Path("new_new_reports_final/f%s_classic_de_evolution_error_dim_%s.csv"%(i, dim))

	error_list = []
	epoch_list = []
	fes_list = []
	error_evolution_list = []
	success_count = 0
	for n in range(n_runs):
		print("Run: %s"%(n))
		de = DE(f, pop_size, n_epochs, dim, bounds, parameters_dict, optimum=optimum, max_obj=False)
		error, success, error_evolution, fes = de.run(show_info=True)
		error_list.append(error)
		fes_list.append(fes)
		error_evolution_list.append(error_evolution)
		success_count+=success

	success_rate = success_count/n_runs

	write_performance("F%s"%(i), dim, error_list, fes_list, fes_list, pop_size, success_rate, savePathReporter, max_obj=False)
	write_parameters_de("F%s"%(i), n_epochs, dim, pop_size, parameters_dict, savePathParameters, max_obj=False)
	write_evolution(dim, error_evolution_list, n_runs, savePathEvolution)
