#=================================================================================================================#
# Author : Benoit Loger
#
# Description : Contain methods to build and solve LSP models
#
# Last update : 21/10/2023 (dd/mm/yyyy)
#
#=================================================================================================================#

"""
	SVC_based(inst::Instance,svc::SVC;time_limit::Int=-1, max_thread::Int=1)

	Build SVC based LSP model (solver CPLEX)
	_ inst : Instance structure
	_ svc : SVC structure of the uncertainty set
	_ time_limit : Optional time limit for CPLEX
	_ max_thread : Optional number of threads used by CPLEX

"""
function SVC_based(inst::Instance,svc::SVC;time_limit::Int=-1, max_thread::Int=1,)
	m = inst.nb_nodes
	n = length(inst.Pert_tab)
	# Initializing the model and the variables
	model = Model(CPLEX.Optimizer)
	@variable(model, x[i in 1:m,j in [j for j in 1:m if i != j]],Bin) # Binary variable : 1 if edge (i,j) is in the solution
	@variable(model, 1 <= r[i in 1:m] <= m) # Position of city i in the solution

	@variable(model,truc >= 0)
	@variable(model,lambda[i in svc.SV,j in 1:n] >= 0)
	@variable(model,mu[i in svc.SV,j in 1:n] >= 0)
	# Objective function and constraints
	@objective(model, Min, sum(sum(x[i,j]*get_dist(inst,i,j) for i in [i for i in 1:inst.nb_nodes if i != j]) for j in 1:inst.nb_nodes) +
	 sum(sum(sum((mu[i,k]-lambda[i,k])*svc.Q[k,j] for k in 1:n) * svc.Data[j,i] for j in 1:n) for i in svc.SV) + truc*svc.Theta)

	@constraint(model,[j in 1:n],sum(sum(svc.Q[j,k]*(lambda[i,k]-mu[i,k]) for k in 1:n) for i in svc.SV) + x[inst.Pert_tab[j][1],inst.Pert_tab[j][2]] == 0)
	@constraint(model,[i in svc.SV, j in 1:n], lambda[i,j] + mu[i,j] == truc*svc.Alpha[i])

	@constraint(model,[j in 1:m], sum(x[i,j] for i in 1:m if i != j) == 1)
	@constraint(model,[i in 1:m], sum(x[i,j] for j in 1:m if i != j) == 1)
	@constraint(model,[i in 1:m, j in [j for j in 2:m if j != i]], r[i] - r[j] + x[i,j]*(m-1) + x[j,i]*(m-3) <= m-2)


	set_optimizer_attribute(model, "CPXPARAM_Threads", max_thread)
	if time_limit > 0
		set_time_limit_sec(model,time_limit)
	end

	return model, length(svc.SV)
end

"""
	solve_model(model::Model;silent=true)

	Solve the model and return the solution
	_ model : Optimization model
	_ silent : Optional. Set the verbosity of CPLEX to 0 if true
"""
function solve_model(model::Model;silent=true)
	if silent set_silent(model) end
	time = (@timed optimize!(model))[2]
	gap = 0
	if termination_status(model) == OPTIMAL
		#x = abs.(value.(model[:x]))
		r = round.(value.(model[:r]))
		solution = order_to_sol(r)

		objective = objective_value(model)
		status="Optimal"
	else
		status="NOTOPT"
		if has_values(model)
			#x = value.(model[:x])
			r = round.(value.(model[:r]))
			solution = order_to_sol(r)
			objective = objective_value(model)
			obj_bound = objective_bound(model)
			gap = round((obj_bound-objective)/obj_bound*100,digits=2)
		else
			status = string("NOSOL")
			objective = -1.0
			x = []
			solution=[]
			gap = 100
		end
	end
	return solution, status, time, gap
end

#===============================================================#
#																#
#           Not important things here just parsing files		#
# 				and saving informations to file        			#
#																#
#===============================================================#

function order_to_sol(r)
	sol = Array{Int,1}(undef,length(r))
	for i in 1:length(r)
		sol[Int(r[i])] = i
	end
	sol[1] = 1
	push!(sol,1)
	return sol
end

function save_sol(solution_path,solution_file,N,nu,status,sol,t,nb_sv,gap)
	sep = ";"
	line = "\n"
	# Open the file
	f = open(string(solution_path,solution_file),"a")
		print(f,N,sep,nu,sep,status,sep,sol,sep,t,sep,nb_sv,sep,gap,line)
	close(f)
end

function load_sol(solution_path,solution_file)
	df = DataFrame(N=Int[],nu=Float64[],status=String[],solution=String[],time=Float64[],sv=Int[],gap=Float64[])
	f = open(string(solution_path,solution_file),"r")
		lines = readlines(f)
		for i in 1:length(lines)
			# println(lines)
			line = Base.split(lines[i],";")
			# println(line)
			res = [parse(Int,line[1]),parse(Float64,line[2]),line[3],line[4],parse(Float64,line[5]),parse(Int,line[6]),parse(Float64,line[7])]
			push!(df,res)
		end
	close(f)
	return(df)
end
