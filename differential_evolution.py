import numpy as np
#import matplotlib.pyplot as plt
import random
import sys, math
#from utils import plot_contourn, plot_fitness
import time
from scipy.stats import cauchy


class DE():
	def __init__(self, function, pop_size, n_epoch, dim, bounds, parameters_dict, optimum, max_obj=False, epsilon=10**(-8),
		show_info=True):
		self.function = function
		self.pop_size = pop_size
		self.n_epoch = n_epoch
		self.dim= dim
		self.min_value, self.max_value = bounds
		self.parameters_dict = parameters_dict
		self.max_obj = max_obj
		self.epsilon = epsilon
		self.optimum = optimum
		self.show_info = show_info
		self.count_fes = 0
		self.maxFes = 10000*dim
		#self.maxFes_list = self.maxFes*(np.array([0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
		self.fes_list = self.maxFes*np.array([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
		self.error_evolution_list = []
		self.nr_diff = self.parameters_dict["nr_diff"]
		self.parameters_dict = parameters_dict
		

	def initialize_settings(self):
		
		self.generate_initial_population = self.generate_random_popupation
		if (self.parameters_dict["cross_method"] == "binomial"):
			self.crossover = self.binomial_crossover

		else:
			print("This crossover method has been not implemented")
			sys.exit()


		if (self.parameters_dict["mutation_method"] == "classic"):
			self.mutation = self.classic_mutation

		elif (self.parameters_dict["mutation_method"] == "variant"):
			self.mutation = self.variant_mutation

		else:
			print("This mutation method has been not implemented")
			sys.exit()

	def generate_random_popupation(self):
		population = []
		for i in range(self.pop_size):
			population.append(np.random.uniform(low=self.min_value, high=self.max_value, size=self.dim))

		return np.array(population)


	def classic_mutation(self, pop_idx, population, best_ind):
		impact_factor = self.parameters_dict["impact_factor"]
		idxs = [idx for idx in range(self.pop_size) if idx!=pop_idx]

		random_vectors = population[np.random.choice(idxs, 2*(self.parameters_dict["nr_diff"])+1, replace=False)]
		vector_base = random_vectors[0]
		vectors_diff = random_vectors[1:]

		if(self.parameters_dict["base"]=="best"):
			vector_base = best_ind

		elif(self.parameters_dict["base"] == "rand-to-best"):
			vector_base = best_ind
			vectors_diff[0] = best_ind

		elif(self.parameters_dict["base"] == "current-to-best"):
			vector_base = population[pop_idx]
			vectors_diff[0] = best_ind
			vectors_diff[0] = population[pop_idx]			

		diff=0
		for i in range(1, len(vectors_diff)):
			#if (self.parameters_dict["impact_factor_mode"] == "random"):
			impact_factor = np.random.uniform(0, 1)

			diff += self.parameters_dict["impact_factor"]*(vectors_diff[i] - vectors_diff[i-1])

		mutated_vector = np.clip(vector_base + diff, self.min_value, self.max_value)

		return np.array(mutated_vector)

	def variant_mutation(self, pop_idx, population, best_ind):
		impact_factor = self.parameters_dict["impact_factor"]
		idxs = [idx for idx in range(self.pop_size) if idx!=pop_idx]

		a, b, c, d, e = population[np.random.choice(idxs, 5, replace=False)]

		if(self.parameters_dict["base"]=="best"):
			a = best_ind

		mutated_vector = np.clip(a + np.random.uniform(0, 1)*(b-c) + np.random.uniform(0, 1)*(d-e), self.min_value, self.max_value)

		return np.array(mutated_vector)


	def binomial_crossover(self, mutant_vector, pop, pop_idx):
		cross_rate = self.parameters_dict["cross_rate"]
		cross_points = np.random.rand(self.dim) < cross_rate
		if not np.any(cross_points):
			cross_points[np.random.randint(0, self.dim)] = True

		trial = np.where(cross_points, mutant_vector, pop[pop_idx])
		trial_fitness = self.evaluate(np.array([trial]))[0]
		return np.array(trial), trial_fitness

	def evaluate(self, X):
		z = []
		for cromo in X:
			z.append(self.function(cromo))
			self.count_fes+=1
			if(self.best_error!=np.inf):
				self.error_evolution_list.append(self.best_error)

		return z


	def run(self, show_info=False):
		self.initialize_settings()
		self.best_error = -np.inf if self.max_obj else np.inf		
		self.count_fes = 0
		population = self.generate_initial_population()
		fitness = self.evaluate(population)                        # evaluate the initial population
		#self.count_fes+=self.pop_size
		best_idx = np.argmax(fitness) if self.max_obj else np.argmin(fitness)
		best_fit = fitness[best_idx]
		best_ind = population[best_idx]
		#if (self.count_fes in self.maxFes_list):
		#error_evolution.append(abs(self.optimum - fitness[best_idx]))
		best_error_miss = True
		fes_final = None		
		epoch = 0
		best_error  = np.inf

		while self.count_fes<=self.maxFes and best_error >  self.epsilon:
			if (show_info):
				print("Epoch: %s, FES: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.count_fes, fitness[best_idx], best_error))

			for i in range(self.pop_size):
				mutated_vector = self.mutation(i, population, best_ind)
				trial_vector, trial_fitness =  self.crossover(mutated_vector,population, i)

				pop_replace, best_replace =self.check_replacement(fitness[i], trial_fitness, fitness[best_idx])

				if (pop_replace):
					fitness[i] = trial_fitness
					population[i] = trial_vector

					if (best_replace):
						best_idx = i
						best_ind = trial_vector

			epoch+=1
			best_error = abs(self.optimum - fitness[best_idx])
			self.best_error = best_error



		self.error_evolution_list.append(self.best_error)
		print("Epoch: %s, FES: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.count_fes, fitness[best_idx], best_error))
		success = 1 if best_error < self.epsilon else 0
		return best_error, success, self.error_evolution_list, self.count_fes

	def check_replacement(self, fit_ind, fit_trial, fit_best):
		pop_replace = fit_trial>fit_ind if self.max_obj else fit_trial  < fit_ind
		best_replace = fit_trial > fit_best if self.max_obj else fit_trial < fit_best
		return pop_replace, best_replace

#FAZER NOVO DE
class CoDE():
	def __init__(self, function, pop_size, n_epoch, dim, bounds, parameters_dict, optimum, max_obj=False, epsilon=10**(-8),
		show_info=True):

		self.function = function
		self.pop_size = pop_size
		self.n_epoch = n_epoch
		self.dim= dim
		self.min_value, self.max_value = bounds
		self.parameters_dict = parameters_dict
		self.max_obj = max_obj
		self.epsilon = epsilon
		self.optimum = optimum
		self.show_info = show_info
		self.count_fes = 0
		self.maxFes = 10000*dim
		self.fes_list = self.maxFes*np.array([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
		self.error_evolution_list = []
		self.archive = []
		self.mean_cross = .5
		self.mean_impact_factor = .5
		self.archive_size = self.parameters_dict['archive_size']

		if (self.parameters_dict['initialization_method'] == 'random'):
			self.generate_initial_population = self.generate_random_population

		elif (self.parameters_dict['initialization_method'] == 'opposite'):
			self.generate_initial_population = self.generate_opposite_population

		self.mutation = self.code_mutation
		self.crossover = self.binomial_crossover

	def generate_opposite_population(self):
		population, fitness  = self.generate_random_population()
		
		opposite_population = self.max_value + self.min_value - population 

		opposite_fitness = self.evaluate(opposite_population)


		final_fitness = np.array(list(fitness) + list(opposite_fitness))
		sorted_fitness_idx = np.argsort(final_fitness)

		final_population = np.concatenate((population, opposite_population))
		final_population = final_population[sorted_fitness_idx]
		final_fitness = final_fitness[sorted_fitness_idx]

		
		return final_population[:self.pop_size], final_fitness[:self.pop_size]

	def generate_random_population(self):	
		population = []
		
		for i in range(self.pop_size):
			population.append(np.random.uniform(low=self.min_value, high=self.max_value, size=self.dim))

		
		fitness = self.evaluate(population)
		return np.array(population), fitness


	def code_mutation(self, pop_idx, population, best_ind, method, nr_diff):
		idxs = [idx for idx in range(self.pop_size) if idx!=pop_idx]

		random_vectors = population[np.random.choice(idxs, 2*(nr_diff)+1, replace=False)]
		vector_base = random_vectors[0]
		vector_diff = random_vectors[1:]
		diff = 0

		if (method == "best"):
			vector_base = best_ind
			#weighted_factor = len(vector_diff)*[self.F]

		elif(method == "current-to-best"):
			vector_base = population[pop_idx]
			diff = self.F*(best_ind - population[pop_idx])

		elif(method == "rand-to-best"):
			vector_base = population[pop_idx]
			diff = np.random.uniform(0, 1)*(best_ind - population[pop_idx])

		for i in range(1, len(vector_diff)):
			diff += self.F*(vector_diff[i-1] - vector_diff[i])

		mutated_vector = np.clip(vector_base + diff, self.min_value, self.max_value)

		return np.array(mutated_vector)

	def evaluate(self, X):
		z = []
		for cromo in X:
			z.append(self.function(np.array(cromo)))
			self.count_fes+=1
			if(self.best_error!=np.inf):
				self.error_evolution_list.append(self.best_error)

		return np.array(z)
			
	def check_replacement(self, fit_ind, fit_trial, fit_best):
		pop_replace = fit_trial>fit_ind if self.max_obj else fit_trial  < fit_ind
		best_replace = fit_trial > fit_best if self.max_obj else fit_trial < fit_best
		return pop_replace, best_replace
	
	def generate_trial_vector(self, pop_idx, population, fitness, best_ind):
		impact_factor = self.parameters_dict["impact_factor"]
		cross_rate = self.parameters_dict['cross_rate']

		# select randomly an configuration for cross_rate and impact factor
		r = np.random.choice(len(impact_factor))
		self.F, self.cr = impact_factor[r], cross_rate[r]

		trial_vector_list = []
		trial_fitness_list = []
		for method, nr_diff in zip(self.parameters_dict["control"], self.parameters_dict["nr_diff"]):
			
			if (method == "current-to-pbest"):
				mut = self.jade_mutation(pop_idx, population, fitness, best_ind)

			else:
				mut = self.code_mutation(pop_idx, population, best_ind, method, nr_diff)
			
			trial_vector, trial_fitness = self.crossover(mut, population, pop_idx, method)
			trial_vector_list.append(trial_vector)
			trial_fitness_list.append(trial_fitness)

		#print(trial_fitness_list)
		select_trial_idx = np.argmax(trial_fitness_list) if self.max_obj else np.argmin(trial_fitness_list)
		#print(select_trial_idx)
		select_trial_vector = np.array(trial_vector_list)[select_trial_idx]
		select_trial_fitness = np.array(trial_fitness_list)[select_trial_idx]
		#print(select_trial_vector, select_trial_fitness)
		return select_trial_vector, select_trial_fitness


	def jade_mutation(self, pop_idx, population, fitness, best_ind):
		
		p = 0.05

		p_best = round(self.pop_size*p) 

		sorted_idx = np.argsort(fitness)
		best_p_fitness = fitness[sorted_idx][:p_best]
		best_p_population = population[sorted_idx][:p_best]

		#print(p_best, best_p_population)
		r = np.random.choice(best_p_population.shape[0])
		best_ind = best_p_population[r]

		idxs = [idx for idx in range(self.pop_size) if idx!=pop_idx]

		random_idx = np.random.choice(idxs, 2, replace=False)

		random_vectors = population[random_idx]
		if (self.archive_size > 0):
			if (len(self.archive) == 0): # during initialization of archive
				population_archive = population
			else:
				#print(np.array(self.archive).shape)
				population_archive = np.concatenate((population, np.array(self.archive)))
			
			idxs.remove(random_idx[0])
			r2_index = np.random.choice(idxs, 1)
			random_vectors[1] = population_archive[r2_index]


		mutated_vector = population[pop_idx] + self.F*(best_ind - population[pop_idx])+self.F*(random_vectors[0]-random_vectors[1])
		mutated_vector = np.clip(mutated_vector, self.min_value, self.max_value)

		return np.array(mutated_vector)

	def binomial_crossover(self, mutant_vector, pop, pop_idx, method):
		#print("entrou binomial")

		#print(mutant_vector)
		#print(pop[pop_idx])
		if (method == "current-to-best"):
			return mutant_vector, self.evaluate(np.array([mutant_vector]))[0]

		else:
			cross_points = np.random.uniform(0, 1, self.dim) < self.cr
			#print(self.cr, cross_points)
			if not np.any(cross_points):
				cross_points[np.random.randint(0, self.dim)] = True

			trial = np.where(cross_points, mutant_vector, pop[pop_idx])
			#print(trial)
			trial_fitness = self.evaluate(np.array([trial]))[0]
			#print(trial_fitness)
			return np.array(trial), trial_fitness

	def opposite_based_evolution(self, population, fitness):
		
		if (np.random.uniform(0, 1) < self.parameters_dict['jumping_rate']):
			max_vector = population.max(axis=0)
			min_vector = population.min(axis=0)
		#	opposite_population = []
			#for i in range(len(population)):
			opposite_population = max_vector*np.ones((self.pop_size, self.dim)) + min_vector*np.ones((self.pop_size, self.dim)) - population
			opposite_population = np.array(opposite_population)
			opposite_fitness = self.evaluate(opposite_population)

			final_fitness = np.array(list(fitness) +  list(opposite_fitness))
			final_population = np.concatenate((population, opposite_population))

			sorted_idx = np.argsort(final_fitness)
			population = final_population[sorted_idx]
			fitness = final_fitness[sorted_idx]

		return population[:self.pop_size], fitness[:self.pop_size]


	def run(self, show_info=True):
		self.archive_fitness = []
		self.best_error = -np.inf if self.max_obj else np.inf		
		self.count_fes = 0
		population, fitness = self.generate_initial_population()
		#fitness = self.evaluate(population)                        # evaluate the initial population
		#self.count_fes+=self.pop_size
		best_idx = np.argmax(fitness) if self.max_obj else np.argmin(fitness)
		best_fit = fitness[best_idx]
		best_ind = population[best_idx]
		#if (self.count_fes in self.maxFes_list):
		#error_evolution.append(abs(self.optimum - fitness[best_idx]))
		best_error_miss = True
		fes_final = None		
		epoch = 0
		best_error  = np.inf

		while self.count_fes<=self.maxFes and best_error >  self.epsilon:
			if (show_info):
				print("Epoch: %s, FES: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.count_fes, fitness[best_idx], best_error))

			set_cross, set_factor = [], []
			for i in range(self.pop_size):
				if ("current-to-pbest" in self.parameters_dict["control"]):
					self.update_control_parameters()
				
				trial_vector, trial_fitness = self.generate_trial_vector(i, population, fitness, best_ind)

				pop_replace, best_replace =self.check_replacement(fitness[i], trial_fitness, fitness[best_idx])

				if (pop_replace):
					fitness[i] = trial_fitness
					population[i] = trial_vector

					self.archive.append(population[i])
					self.archive_fitness.append(fitness[i])
					set_cross.append(self.cr)
					set_factor.append(self.F)

					if (best_replace):
						best_idx = i
						best_ind = trial_vector


			if (self.parameters_dict["initialization_method"] == "opposite"):
				population, fitness = self.opposite_based_evolution(population, fitness)
				best_idx = np.argmax(fitness) if self.max_obj else np.argmin(fitness)
				best_ind = population[best_idx]

			if ("current-to-pbest" in self.parameters_dict["control"]):
				self.remove_archive()
				self.update_adaptive_parameters(set_cross, set_factor)


			epoch+=1
			best_error = abs(self.optimum - fitness[best_idx])
			self.best_error = best_error

		self.error_evolution_list.append(self.best_error)
		print("Epoch: %s, FES: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.count_fes, fitness[best_idx], best_error))
		success = 1 if best_error < self.epsilon else 0
		return best_error, success, self.error_evolution_list, self.count_fes

	def remove_archive(self):
		if (len(self.archive) > self.pop_size):
			if (self.parameters_dict["remove_method"]=="random"):
				nr_remove_pop = len(self.archive) - self.pop_size
				idx_to_remove = np.random.choice(list(range(len(self.archive))), nr_remove_pop)
				self.archive = list(np.delete(self.archive, idx_to_remove, axis=0))

			else:
				nr_remove_pop = len(self.archive) - self.pop_size
				nr_remove_random = nr_remove_pop//2

				idx_sort = np.argsort(self.archive_fitness)
				sorted_archive = np.array(self.archive)[idx_sort]
				sorted_fitness = np.array(self.archive_fitness)[idx_sort]

				self.archive = sorted_archive[:(self.pop_size+nr_remove_random)]
				self.archive_fitness = sorted_fitness[:(self.pop_size+nr_remove_random)]
				idx_to_remove = np.random.choice(list(range(len(self.archive))), nr_remove_pop)
				self.archive = list(np.delete(self.archive, idx_to_remove, axis=0))
				self.archive_fitness = list(np.delete(self.archive_fitness, idx_to_remove, axis=0))





	def update_control_parameters(self):
		self.cr = np.clip(np.random.normal(self.mean_cross, .1), 0, 1)
		self.F = None

		while (self.F is None):
			f_temp = cauchy.rvs(loc=self.mean_impact_factor, scale=.1, size=1)[0]  # draw 1 samples according cauchy dist

			if (f_temp > 0):
				self.F = f_temp


	def update_adaptive_parameters(self, set_cross, set_factor):
		c = 0.1
		self.mean_cross = (1-c)*self.mean_cross + c*np.mean(set_cross)

		num = 0
		den = 0

		if (len(set_factor)!=0):
			for f in set_factor:
				num += f**2
				den += f

			lehmmen_mean = num/den
			self.mean_impact_factor = (1-c)*self.mean_impact_factor + c*lehmmen_mean





class JADE(CoDE):
	def __init__(self, function, pop_size, n_epoch, dim, bounds, parameters_dict, optimum, max_obj=False, epsilon=10**(-8)):

		super().__init__( function, pop_size, n_epoch, dim, bounds, parameters_dict, optimum)

		self.archive = [] 
		self.mean_cross = self.parameters_dict["cross_rate"]
		self.mean_impact_factor = self.parameters_dict["impact_factor"]
		self.archive_size = self.parameters_dict["archive_size"]


	def update_control_parameters(self):
		self.cr = np.clip(np.random.normal(self.mean_cross, .1), 0, 1)
		self.F = None

		while (self.F is None):
			f_temp = cauchy.rvs(loc=self.mean_impact_factor, scale=.1, size=1)[0]  # draw 1 samples according cauchy dist

			if (f_temp > 0):
				self.F = f_temp

	def jade_mutation(self, pop_idx, population, fitness, best_ind):
		
		p = 0.05

		p_best = round(self.pop_size*p) 

		sorted_idx = np.argsort(fitness)
		best_p_fitness = fitness[sorted_idx][:p_best]
		best_p_population = population[sorted_idx][:p_best]

		#print(p_best, best_p_population)
		r = np.random.choice(best_p_population.shape[0])
		best_ind = best_p_population[r]

		idxs = [idx for idx in range(self.pop_size) if idx!=pop_idx]

		random_idx = np.random.choice(idxs, 2*(self.parameters_dict['nr_diff']), replace=False)

		random_vectors = population[random_idx]
		if (self.archive_size > 0):
			if (len(self.archive) == 0): # during initialization of archive
				population_archive = population
			else:
				#print(np.array(self.archive).shape)
				population_archive = np.concatenate((population, np.array(self.archive)))
			
			idxs.remove(random_idx[0])
			r2_index = np.random.choice(idxs, 1)
			random_vectors[1] = population_archive[r2_index]


		mutated_vector = population[pop_idx] + self.F*(best_ind - population[pop_idx])+self.F*(random_vectors[0]-random_vectors[1])
		mutated_vector = np.clip(mutated_vector, self.min_value, self.max_value)

		return np.array(mutated_vector)


	def binomial_crossover(self, mutant_vector, pop, pop_idx):

		cross_points = np.random.uniform(0, 1, self.dim) < self.cr

		if not np.any(cross_points):
			cross_points[np.random.randint(0, self.dim)] = True

		trial = np.where(cross_points, mutant_vector, pop[pop_idx])
		trial_fitness = self.evaluate(np.array([trial]))[0]

		return np.array(trial), trial_fitness

	def remove_archive(self):
		if (len(self.archive) > self.pop_size):
			nr_remove_pop = len(self.archive) - self.pop_size
			idx_to_remove = np.random.choice(list(range(len(self.archive))), nr_remove_pop)
			self.archive = list(np.delete(self.archive, idx_to_remove, axis=0))

	def update_adaptive_parameters(self, set_cross, set_factor):
		c = 0.1
		self.mean_cross = (1-c)*self.mean_cross + c*np.mean(set_cross)

		num = 0
		den = 0

		if (len(set_factor)!=0):
			for f in set_factor:
				num += f**2
				den += f

			lehmmen_mean = num/den

			self.mean_impact_factor = (1-c)*self.mean_impact_factor + c*lehmmen_mean



	def run(self, show_info=True):
		self.best_error = -np.inf if self.max_obj else np.inf		
		self.count_fes = 0
		population, fitness = self.generate_initial_population()

		best_idx = np.argmax(fitness) if self.max_obj else np.argmin(fitness)
		best_ind = population[best_idx]
		epoch = 0
		best_error  = np.inf

		while self.count_fes<=self.maxFes and best_error >  self.epsilon:
			if (show_info):
				print("Epoch: %s, FES: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.count_fes, fitness[best_idx], best_error))

			set_factor, set_cross = [], []

			for i in range(self.pop_size):
				self.update_control_parameters()
				mutated_vector = self.jade_mutation(i, population, fitness, best_ind)
				trial_vector, trial_fitness = self.binomial_crossover(mutated_vector, population, i)

				if (trial_fitness < fitness[i]):
				
					fitness[i] = trial_fitness
					population[i] = trial_vector
					#print(type(self.archive))
					self.archive.append(population[i])
					set_cross.append(self.cr)
					set_factor.append(self.F)


			self.remove_archive()
			self.update_adaptive_parameters(set_cross, set_factor)

			epoch+=1
			best_error = abs(self.optimum - fitness[best_idx])
			self.best_error = best_error

		self.error_evolution_list.append(self.best_error)
		print("Epoch: %s, FES: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.count_fes, fitness[best_idx], best_error))
		success = 1 if best_error < self.epsilon else 0
		return best_error, success, self.error_evolution_list, self.count_fes





