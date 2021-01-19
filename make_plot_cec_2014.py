import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, sys
from pathlib import Path





fontsize = 16
de_color = "steelblue"
code_color = "salmon"
jade_color = "black"
pso_color = "darkviolet"
jacode_color = "goldenrod"


f_list = [1, 2, 4, 6, 7, 9, 14]
dim = 10
if (dim == 10):
	fes = [100, 1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 99999]
else:
	fes = [100, 1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 99999, 200000,
	210000, 220000, 230000, 250000, 260000, 270000, 280000, 290000]

for f in f_list:
	print(f)
	df_de = pd.read_csv("new_new_reports_final/f%s_classic_de_evolution_error_dim_%s.csv"%(f, dim))
	df_code = pd.read_csv("new_new_reports_final/f%s_CODE_evolution_error_dim_%s.csv"%(f, dim))
	df_jade = pd.read_csv("new_new_reports_final/f%s_JADE_evolution_error_dim_%s.csv"%(f, dim))
	df_pso = pd.read_csv("new_new_reports_final/f%s_pso_evolution_error_dim_%s.csv"%(f, dim))
	df_jacode = pd.read_csv("new_new_reports_final/alternative_f%s_JACODE_evolution_error_dim_%s.csv"%(f, dim))


	save_path_png = Path("new_new_reports_final/plots/f%s_dim_%s_mean.png"%(f, dim))
	save_path_pdf = Path("new_new_reports_final/plots/f%s_dim_%s_mean.pdf"%(f, dim))

	#de = df_de[(df_de.Fi==f) & (df_de.dim==dim)]
	de = np.median(np.array([df_de.iloc[:, i].values[fes] for i in range(1, df_de.shape[1])]), axis=0)

	#pso = df_pso[(df_pso.Fi==f) & (df_pso.dim==dim)]
	pso = np.median(np.array([df_pso.iloc[:, i].values[fes] for i in range(1, df_pso.shape[1])]), axis=0)
		
	#code = df_code[(df_code.Fi==f) & (df_code.dim==dim)]
	code = np.median(np.array([df_code.iloc[:, i].values[fes] for i in range(1, df_code.shape[1])]), axis=0)

	#jade = df_jade[(df_jade.Fi==f) & (df_jade.dim==dim)]
	jade = np.median(np.array([df_jade.iloc[:, i].values[fes] for i in range(1, df_jade.shape[1])]), axis=0)

	#jacode = df_jacode[(df_jacode.Fi==f) & (df_jacode.dim==dim)]
	jacode = np.median(np.array([df_jacode.iloc[:, i].values[fes] for i in range(1, df_jacode.shape[1])]), axis=0)

	fig, ax = plt.subplots()
	plt.plot(np.array(fes)/fes[-1], de, marker="x", color=de_color, label="DE")
	plt.plot(np.array(fes)/fes[-1], code, marker="o", color=code_color, label="CoDE")
	plt.plot(np.array(fes)/fes[-1], jade, marker="d", color=jade_color, label="JADE")
	plt.plot(np.array(fes)/fes[-1], pso, marker="p", color=pso_color, label="PSO")
	plt.plot(np.array(fes)/fes[-1], jacode, marker="*", color=jacode_color, label="JACODE")
	plt.plot(np.array(fes)/fes[-1], (10**(-8))*np.ones(len(fes)), marker="s", color="red", label="Limiar de Sucesso")
		
	plt.legend(frameon=False, fontsize=fontsize-2, ncol=2)
	plt.xlabel("Fes/MaxFes", fontsize=fontsize+4)
	plt.ylabel("Mediana do Erro", fontsize=fontsize+4)
	plt.xticks(fontsize=fontsize+4)
	plt.yticks(fontsize=fontsize+4)
	#plt.xlim(0, 100000)
	plt.ylim(10**(-9), 10**9)
	plt.yscale("log")
	plt.tight_layout()
	plt.savefig(save_path_png)
	plt.savefig(save_path_pdf)