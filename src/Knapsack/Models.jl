#=================================================================================================================#
# Author : Benoit Loger
#
# Description : Knapsack optimization model
#
# Last update : 21/10/2023 (dd/mm/yyyy)
#
#=================================================================================================================#

"""
	SVC_based(inst::Instance,svc::SVC;time_limit::Int=-1, max_thread::Int=1,variable_type="Binary")

	Build SVC based Knapsack model (solver CPLEX)
	_ inst : Instance structure
	_ svc : SVC structure of the uncertainty set
	_ time_limit : Optional time limit for CPLEX
	_ max_thread : Optional number of threads used by CPLEX
	_ variable_type : Select the type of variables (e.g. "Continuous", default = "Binary")

"""
function SVC_based(inst::Instance,svc::SVC;time_limit::Int=-1, max_thread::Int=1,variable_type="Binary")
	m = length(inst.c)
	# Initialize model
	model = Model(CPLEX.Optimizer)

	# Initialize variables
	if variable_type == "Binary"
		@variable(model, x[i in 1:m] >= 0,Bin)
	elseif variable_type == "Integer"
		@variable(model, x[i in 1:m] >= 0,Int)
	elseif variable_type == "Continuous"
		@variable(model, 1 >= x[i in 1:m] >= 0)
	end
	@variable(model,truc >= 0)
	@variable(model,lambda[i in svc.SV,j in 1:m] >= 0)
	@variable(model,mu[i in svc.SV,j in 1:m] >= 0)

	@objective(model, Max, sum(inst.c[i]*x[i] for i in 1:m))
	@constraint(model,sum(sum(sum((mu[i,k]-lambda[i,k])*svc.Q[k,j] for k in 1:m) * svc.Data[j,i] for j in 1:m) for i in svc.SV) + truc*svc.Theta <= inst.b)
	@constraint(model,[j in 1:m],sum(sum(svc.Q[j,k]*(lambda[i,k]-mu[i,k]) for k in 1:m) for i in svc.SV) + x[j] == 0)
	@constraint(model,[i in svc.SV, j in 1:m], lambda[i,j] + mu[i,j] == truc*svc.Alpha[i])

	set_optimizer_attribute(model, "CPXPARAM_Threads", max_thread)
	if time_limit > 0
		set_time_limit_sec(model,time_limit)
	end
	return model, length(svc.SV)
end


"""
	solve_model(model::Model;silent=true)

	Solve the model and return selected items
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
	return round.(x,digits=3), status, time, gap
end


"""
	evaluate(inst::Instance,sol::Array{Float64,1},w_test::Array{Float64,2})

	Evaluate the objective function and constraint satisfaction of a solution
	_ inst : Instance structure
	_ sol : Solutions to evaluate
	_ w_test : Set of test weights
"""
function evaluate(inst::Instance,sol::Array{Float64,1},w_test::Array{Float64,2})
	m = length(sol)
	N = size(w_test,2)
	obj = sum(inst.c[i]*sol[i] for i in 1:m)
	weights = [sum(w_test[i,j]*sol[i] for i in 1:m) for j in 1:N]
	load = weights./inst.b.*100
	return obj, count(e->e<=inst.b,weights)/N, load
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
