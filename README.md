# ApproximatedKernelLearningRO
This repository contain the code and data used in the paper: Approximate Kernel Learning Uncertainty Set for Robust Combinatorial Optimization (Add link when published)

All the code used to generate uncertainty sets, build and solve optimization models is available in the src folder. The data and instances used in the experiments are available in the Instance folder.
The code is well commented and easy to understand. The following julia function show how to execute our code for the Knapsack problem in the the setting presented in our experimental results. Similar examples are available for the LotSizing and the Traveling Salesman problems in the corresponding folders.

```julia
function Knapsack()
	name="gamma_10" # tpye of distribution and number of items
	i = 1 # ID of the instance (from 1 to 10)
	nu = 0.15 # Value of SVC parameter \nu
	K = 30 # Number of support vector selected by our method

	inst = Instance(name,i,path=INSTANCE_PATH) # Loading the instance
	w_test = inst.dist.generate(10000) # Generate a test_set of 10 000 weights scenarios

	# Creating initial cluster
	svc = SVC(inst.w,nu) # SVC structure

	# If you need to build the cluster
	svc.learn()	# Learning
	svc.write(string(CLUSTER_PATH,name,"SVC/SVC_$(i)_$(nu).txt")) # Saving cluster

	# If you need to read the cluster
	svc.write(string(CLUSTER_PATH,name,"SVC/SVC_$(i)_$(nu).txt"))

	# Compute the initial coverage
	U_ini, cov_ini = svc.coverage(w_test)
	println(cov_ini)

	# If you need to execute phase 1
	app, groups, max_distance, LP_time = approximation(svc,min(K,length(setdiff(svc.SV,svc.BSV))))
	app.write(string(CLUSTER_PATH,name,"ASVC*/ASVC*_$(nu)_$(K)_$(i).txt")) # Saving approximated cluster

	# If you need to read the result of phase 1
	app = SVC(inst.w,nu)
	app.read(string(CLUSTER_PATH,name,"ASVC*/ASVC*_$(nu)_$(K)_$(i).txt"))

	# Compute coverage after phase 1
	U_p1, cov_p1 = app.coverage(w_test)
	println(cov_p1)

	# If you need to execute phase 2
	app_phase_2, max_distance, MIP_time, LP_time = phase_2(svc,app,time_limit=3600) # execute phase 2 during 3600 seconds
	app_phase_2.write(string(CLUSTER_PATH,name,"ASVC/ASVC_$(nu)_$(K)_$(i).txt")) # Saving approximated cluster

	# If you need to read the result of phase 2
	app_phase_2 = SVC(inst.w,nu)
	app_phase_2.read(string(CLUSTER_PATH,name,"ASVC/ASVC_$(nu)_$(K)_$(i).txt"))

	# Compute coverage after phase 2
	U_p2, cov_p2 = app_phase_2.coverage(w_test)
	println(cov_p2)

	# Building and solve Knapsack model for the initial uncertainty set
	svc_model, nb_sv = SVC_based(inst,svc,variable_type="Binary",time_limit=36000,max_thread=1)
	println(" * Solving ")
	svc_sol, svc_status, svc_t, svc_gap = solve_model(svc_model,silent=false)
	println(" * Model solved in $svc_t seconds")
	println(" * Saving solution to file")
	save_sol(string(SOLUTION_PATH,name),"SVC_$(i).txt",length(inst.w[1,:]),nu,svc_status,string(svc_sol),svc_t,nb_sv,svc_gap)

	# Building and solve Knapsack model for the initial uncertainty set
	asvc_model, nb_sv = SVC_based(inst,app_phase_2,variable_type="Binary",time_limit=36000,max_thread=1)
	println(" * Solving ")
	asvc_sol, asvc_status, asvc_t, asvc_gap = solve_model(asvc_model,silent=false)
	println(" * Model solved in $asvc_t seconds")
	println(" * Saving solution to file")
	save_sol(string(SOLUTION_PATH,name),"ASVC_$(K)_$(i).txt",length(inst.w[1,:]),nu,asvc_status,string(asvc_sol),asvc_t,nb_sv,asvc_gap)
end
```
