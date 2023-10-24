#=================================================================================================================#
# Author : Benoit Loger
#
# Description : Contain a test function showing how to execute usefull functions
#
# Last update : 21/10/2023 (dd/mm/yyyy)
#
#=================================================================================================================#
using JuMP, CPLEX, Distributions, LinearAlgebra, DataFrames, CSV, Clustering

include("../SVC_struct.jl")
include("../var.jl")
include("../Dist.jl")
include("Instance.jl")
include("Models.jl")

INSTANCE_PATH = "../../Instances/LSP/"
CLUSTER_PATH = "../../Clusters/LSP/"
SOLUTION_PATH = "../../Solutions/LSP/"

"""
	Illustrating example generating SVC cluster, approximations (phase 1 and 2) and solving the corresponding LSP model
"""
function test()
	# Load the instance
	name="test/"
	nu = 0.15
	K = 56
	inst = Instance(name,path=INSTANCE_PATH) # Loading the instance
	d_test = Matrix{Float64}(DataFrame(CSV.File(string(INSTANCE_PATH,name,"demandsTest.csv")))) # Load a test_set of 10 000 weights scenarios

	# Creating initial cluster
	svc = SVC(Float64.(inst.W),nu) # SVC structure

	# If you need to build the cluster
	svc.learn()	# Learning
	svc.write(string(CLUSTER_PATH,name,"SVC/SVC_$(nu).txt")) # Saving cluster

	# If you need to read the cluster
	svc.write(string(CLUSTER_PATH,name,"SVC/SVC_$(nu).txt"))

	# Compute the initial coverage
	U_ini, cov_ini = svc.coverage(d_test)
	println(cov_ini)

	# If you need to execute phase 1
	app, groups, max_distance, LP_time = approximation(svc,min(K,length(setdiff(svc.SV,svc.BSV))))
	app.write(string(CLUSTER_PATH,name,"ASVC*/ASVC*_$(nu)_$(K).txt")) # Saving approximated cluster

	# If you need to read the result of phase 1
	app = SVC(Float64.(inst.W),nu)
	app.read(string(CLUSTER_PATH,name,"ASVC*/ASVC*_$(nu)_$(K).txt"))

	# Compute coverage after phase 1
	U_p1, cov_p1 = app.coverage(d_test)
	println(cov_p1)

	# If you need to execute phase 2
	app_phase_2, max_distance, MIP_time, LP_time = phase_2(svc,app,time_limit=10) # execute phase 2 during 3600 seconds
	app_phase_2.write(string(CLUSTER_PATH,name,"ASVC/ASVC_$(nu)_$(K).txt")) # Saving approximated cluster

	# If you need to read the result of phase 2
	app_phase_2 = SVC(Float64.(inst.W),nu)
	app_phase_2.read(string(CLUSTER_PATH,name,"ASVC/ASVC_$(nu)_$(K).txt"))

	# Compute coverage after phase 2
	U_p2, cov_p2 = app_phase_2.coverage(d_test)
	println(cov_p2)

	# Building and solve LSP model for the initial uncertainty set
	svc_model, nb_sv = SVC_based(inst,svc,time_limit=10,max_thread=1)
	println(" * Solving ")
	svc_sol, svc_status, svc_t, svc_gap = solve_model(svc_model,silent=false)
	println(" * Model solved in $svc_t seconds")
	println(" * Saving solution to file")
	save_sol(string(SOLUTION_PATH,name),"SVC.txt",length(inst.W[1,:]),nu,svc_status,string(svc_sol),svc_t,nb_sv,svc_gap)

	# Building and solve LSP model for the initial uncertainty set
	asvc_model, nb_sv = SVC_based(inst,app_phase_2,time_limit=10,max_thread=1)
	println(" * Solving ")
	asvc_sol, asvc_status, asvc_t, asvc_gap = solve_model(asvc_model,silent=false)
	println(" * Model solved in $asvc_t seconds")
	println(" * Saving solution to file")
	save_sol(string(SOLUTION_PATH,name),"ASVC_$(K).txt",length(inst.W[1,:]),nu,asvc_status,string(asvc_sol),asvc_t,nb_sv,asvc_gap)
end
