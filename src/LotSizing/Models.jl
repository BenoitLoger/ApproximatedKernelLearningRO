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
	_ variable_type : Select the type of variables (e.g. "Continuous", default = "Binary")

"""
function SVC_based(inst::Instance,svc::SVC;max_thread=1,time_limit=3600)
	d = [mean(inst.W[k,:]) for k in 1:inst.T]

	model = Model(CPLEX.Optimizer)

	#
	@variable(model,x[k in 1:inst.T] >= 0) # Quantity ordered at period k
	@variable(model,v[k in 1:inst.T], Bin) # 1 if positive ordered quantity at period k
	@variable(model, y[k in 1:inst.T] >= 0) # Backlogg and shortage cost at period k


	# Objective function
	@objective(model,Min,sum(inst.C*x[k] + inst.K*v[k] + y[k] for k in 1:inst.T))

	# Binary constraint
	@constraint(model,[t in 1:inst.T], x[t] <= v[t] * sum(d))

	# SVC based constraints
	@variable(model,eta_max[t in 1:inst.T] >= 0)
	@variable(model,lambda_max[t in 1:inst.T, i in svc.SV,k in 1:t] >= 0)
	@variable(model,mu_max[t in 1:inst.T, i in svc.SV,k in 1:t] >= 0)

	@constraint(model,[t in 1:inst.T],y[t] >= -inst.P*( 0 + sum( x[k] for k in 1:t) - (sum(sum(sum((mu_max[t,i,k]-lambda_max[t,i,k])*svc.Q[k,j] for k in 1:t) * svc.Data[j,i] for j in 1:t) for i in svc.SV) + eta_max[t]*svc.Theta)))
	@constraint(model,[t in 1:inst.T,j in 1:t],sum(sum(svc.Q[j,k]*(lambda_max[t,i,k]-mu_max[t,i,k]) for k in 1:t) for i in svc.SV) + 1 == 0)
	@constraint(model,[t in 1:inst.T,i in svc.SV, j in 1:t], lambda_max[t,i,j] + mu_max[t,i,j] == eta_max[t]*svc.Alpha[i])


	@variable(model,eta_min[t in 1:inst.T] <= 0)
	@variable(model,lambda_min[t in 1:inst.T, i in svc.SV,k in 1:t] <= 0)
	@variable(model,mu_min[t in 1:inst.T, i in svc.SV,k in 1:t] <= 0)

	@constraint(model,[t in 1:inst.T],y[t] >= inst.H*( 0 + sum( x[k] for k in 1:t) - (sum(sum(sum((mu_min[t,i,k]-lambda_min[t,i,k])*svc.Q[k,j] for k in 1:t) * svc.Data[j,i] for j in 1:t) for i in svc.SV) + eta_min[t]*svc.Theta)))
	@constraint(model,[t in 1:inst.T,j in 1:t],sum(sum(svc.Q[j,k]*(lambda_min[t,i,k]-mu_min[t,i,k]) for k in 1:t) for i in svc.SV) + 1 == 0)
	@constraint(model,[t in 1:inst.T,i in svc.SV, j in 1:t], lambda_min[t,i,j] + mu_min[t,i,j] == eta_min[t]*svc.Alpha[i])

	# COnfigure solver to avoid the generation of cuts (useless and takes a lot of time)
	set_optimizer_attribute(model, "CPXPARAM_Threads", max_thread)
	set_optimizer_attribute(model,"CPXPARAM_MIP_Cuts_FlowCovers",-1)
	set_optimizer_attribute(model, "CPXPARAM_MIP_Cuts_MIRCut",-1)
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
		x = abs.(value.(model[:x]))
		objective = objective_value(model)
		status="Optimal"
	else
		status="NOTOPT"
		if has_values(model)
			x = value.(model[:x])
			objective = objective_value(model)
			obj_bound = objective_bound(model)
			gap = round((obj_bound-objective)/obj_bound*100,digits=2)
		else
			status = string("NOSOL")
			objective = -1.0
			x = []
			gap = 100
		end
	end
	println(objective)
	return round.(x,digits=3), status, time, gap, objective
end

"""
	evaluate(inst::Instance,sol::Array{Float64,1},w_test::Array{Float64,2})

	Evaluate the real cost of a solution on a test set
	_ inst : Instance structure
	_ sol : Solutions to evaluate
	_ w_test : Set of test weights
"""
function evaluate(inst::Instance,sol,testSet::Matrix{Float64})
	fixed_cost = count(e->e>0,sol) * inst.K
	unit_cost = sum(sol)* inst.C
	costs = Matrix{Float64}(undef,size(testSet))
	for j in 1:length(testSet[:,1])
		for i in 1:length(testSet[1,:])
			costs[j,i] = sum( max(inst.H*sum(sol[k]-testSet[k,i] for k in 1:t),-inst.P*sum(sol[k]-testSet[k,i] for k in 1:t)) for t in 1:j)
		end
	end
	return costs
end

#===============================================================#
#																#
#           Not important things here just parsing files		#
# 				and saving informations to file        			#
#																#
#===============================================================#
function save_sol(solution_path,solution_file,N,nu,status,sol,t,nb_sv,gap)
	sep = ";"
	line = "\n"
	# Open the file
	f = open(string(solution_path,solution_file),"a")
		print(f,N,sep,nu,sep,status,sep,sol,sep,t,sep,nb_sv,sep,gap,line)
	close(f)
end

function save_complete_sol(solution_path,solution_file,N,nu,status,sol,y,t,gap,obj)
	sep = ";"
	line = "\n"
	# Open the file
	println("ui")
	f = open(string(solution_path,solution_file),"a")
		print(f,N,sep,nu,sep,status,sep,sol,sep,y,sep,t,sep,gap,sep,obj,line)
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

function load_complete_sol(solution_path,solution_file)
	df = DataFrame(N=Int[],nu=Float64[],status=String[],solution=String[],y=String[],time=Float64[],gap=Float64[],obj=Float64[])
	f = open(string(solution_path,solution_file),"r")
		lines = readlines(f)
		for i in 1:length(lines)
			# println(lines)
			line = Base.split(lines[i],";")
			# println(line)
			res = [parse(Int,line[1]),parse(Float64,line[2]),line[3],line[4],line[5],parse(Float64,line[6]),parse(Float64,line[7]),parse(Float64,line[8])]
			push!(df,res)
		end
	close(f)
	return(df)
end
