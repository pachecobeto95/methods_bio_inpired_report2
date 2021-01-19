import random
import math
import numpy as np, os, sys
#from optproblems import cec2005
from particle_swarm import PSO
from pathlib import Path
from utils import write_parameters_pso, write_performance, write_evolution

import sys
sys.path.append("/home/pacheco/cec2014/python")
#import cec2014
from cec2014 import cec14

class FuncObj():
	def __init__(self, i):
		self.i = i

	def func(self, arr):
		return cec14(arr, self.i)


dim = 30
bounds = [-100, 100]
method_init = "zero"

#optimum_list = np.array([100, 200, 400, 600, 700, 900, 1400])
#f_id_list = optimum_list/100
optimum_list = np.array([1400])
f_id_list = optimum_list/100

#function_list = [FuncObj(f_id).func for f_id in f_id_list]
function_list = [FuncObj(14).func]

n_runs = 25
pop_size = 100
n_epochs = 100000

inertia_max = 0.9
inertia_min = 0.2
cognitive_const = 2.05
social_const = 2.05
parameters_dict = {"method_init": "zero", "inertia":inertia_min, "cognitive_const":cognitive_const, 
"social_const":social_const}

best_error_list = []
fes_list = []
error_evolution_list = []
success_count = 0
savePathParameters = Path("new_new_reports_final/pso_parameters.csv")
savePathReporter = Path("new_new_reports_final/pso_statistics_%s.csv"%(dim))

for i, f, optimum in zip(f_id_list, function_list, optimum_list):
	savePathEvolution = Path("new_new_reports_final/f%s_pso_evolution_error_dim_%s.csv"%(int(i), dim))

	for n in range(n_runs):
		print("Runs: %s"%(n))
		ps = PSO(f, pop_size, n_epochs, dim, bounds, parameters_dict, optimum=optimum)
		best_error, success, error_evolution, fes = ps.run(show_info=True)
		best_error_list.append(best_error)
		error_evolution_list.append(error_evolution)
		fes_list.append(fes)
		success_count+=success

	success_rate = success_count/n_runs
	print("Success Rate: %s"%(success_rate))

	write_performance("F%s"%(i), dim, best_error_list, fes_list, fes_list, pop_size, success_rate, savePathReporter, max_obj=False)
	write_parameters_pso("F%s"%(i), n_epochs, dim, pop_size, parameters_dict, savePathParameters, max_obj=False)
	write_evolution(dim, error_evolution_list, n_runs, savePathEvolution)
