#=================================================================================================================#
# Author : Benoit Loger
#
# Description : Contain methods related to Support Vector Clustering and its approximation
#
# Last update : 01/06/2022 (dd/mm/yyyy)
#
#=================================================================================================================#
using JuMP, LinearAlgebra, Statistics, Distributions, CPLEX
"""
	Structure of Cluster (built from SVC)
"""
mutable struct SVC
	# List of attributes
	Data::Matrix{Float64} # Learning data-set
	Q::Matrix{Float64} # Weighting matrix
	nu::Float64 # Portion of outliers 0 < nu < 1
	Alpha::Vector{Float64} # Weights of each sample point
	SV::Vector{Int} # Index of support vectors
	BSV::Vector{Int} # Index of boundary support vector
	Theta::Float64 # Bound of the uncertainty set

	# List of methods (fake Object Oriented Programming)
	learn::Function # Apply SVC on Data
	write::Function # Write the results in a file
	read::Function # Read SVC results from a file
	is_inside::Function # Check if a point is inside the cluster
	coverage::Function # Check the portion of a Test data-set covered by the cluster
	value::Function # compute Theta^{(i)}

	function SVC(learning_set::Matrix{Float64},nu::Float64)
		Q = weighting_matrix(learning_set)
		this = new(learning_set,Q,nu)

		# Implement learning function (solve QP and generate SV, BSV sets)
		this.learn = function()
			this.Alpha, this.SV, this.BSV = compute_SVC(this.Q,this.Data,this.nu)
			this.Theta = round(minimum([sum(this.Alpha[i]*norm(this.Q*(this.Data[:,ii]-this.Data[:,i]),1) for i in this.SV) for ii in this.BSV]),digits=10)
		end

		# Check if a point is inside of the cluster
		this.is_inside = function(point::Vector{Float64})
			return sum(this.Alpha[i]*norm(this.Q*(point-this.Data[:,i]),1) for i in this.SV) <= this.Theta
		end

		# Return the value of theta evaluated for a point
		this.value = function(point::Vector{Float64})
			return sum(this.Alpha[i]*norm(this.Q*(point-this.Data[:,i]),1) for i in this.SV)
		end

		# Return the indices of the subset of Test covered by the SVC and the portion
		this.coverage = function(Test::Matrix{Float64})
			N = length(Test[1,:])
			covered = [i for i in 1:N if this.is_inside(Test[:,i])]
			portion = length(covered)/N
			return covered, portion
		end

		# save the SVC object to a file
		this.write = function(file)
			return write(file,this.Alpha,this.SV,this.BSV,this.Theta)
		end

		# Read the SVC object from a file
		this.read = function(file)
			this.Q = weighting_matrix(this.Data)
			this.Alpha, this.SV, this.BSV, this.Theta, time = read(file)
			return time
		end

		return this
	end
end



"""
	weighting_matrix(set::Array{Float64,2})

	Compute the weighting matrix Q used in SVC with Weighted General Intersection Kernel (WGIK)
	_ set : Learning data-set
"""
function weighting_matrix(set::Array{Float64,2})
	nb_sample = size(set,2)
	cov_matrix = cov(set,dims=2)
	weighting_matrix = Array{Float64,2}(cov_matrix^(-1/2))

	return weighting_matrix
end

function deci(x::Float64)
   i = 1
   while true
       x = x*10
       if x >= 1
           return i
       end
       i += 1
   end
end

"""
	compute_SVC(Q::Array{Float64,2},set::Array{Float64,2},nu::Float64; verbose::Bool = false,round_factor::Int=8, epsilon::Float64=0.01)

	Compute the value of alpha and constructs the sets SV and BSV of the SVC

	* Params :
		_ Q : Weighting matrix
		_ set : Learning data-set
		_ nu : Control the size of the cluster 0 < nu < 1
"""
function compute_SVC(Q::Array{Float64,2},set::Array{Float64,2},nu::Float64; verbose::Bool = false,round_factor::Int=8, epsilon::Float64=0.01)
	nb_sample = size(set,2)
	nb_column = size(set,1)
	N = size(set,2)

	# First steps to compute the Kernel matrix
	weighted_tab = [[transpose(Q[line,:])*set[:,sample] for sample in 1:nb_sample] for line in 1:nb_column]
	lk = [maximum(weighted_tab[column])-minimum(weighted_tab[column])+epsilon for column in 1:nb_column]
	lk = sum(lk[column] for column in 1:nb_column)

	# Building model
	m = Model(CPLEX.Optimizer) # Initialise model
	# Solver parameter
	if !verbose
		set_silent(m) # No output
	end
	set_optimizer_attribute(m,"CPXPARAM_OptimalityTarget",2)		# Looking for global optima
	set_optimizer_attribute(m,"CPXPARAM_Emphasis_Numerical",true) 	# Emphasis numerical precision
	set_optimizer_attribute(m, "CPXPARAM_Threads", 1) 				# Maximum number of threads

	# Decision variables
	@variable(m,0 <= alpha[i in 1:N] <= 1/(N*nu))

	# Objective function
	@objective(m, Min, sum( sum(alpha[i]*alpha[j]*(lk-norm(Q*(set[:,i]-set[:,j]),1)) for j in 1:N ) for i in 1:N) - sum( alpha[i]*(lk-norm(Q*(set[:,i]-set[:,i]),1)) for i in 1:N))

	# Constraints
	@constraint(m, sum_eq_1,sum(alpha[i] for i in 1:N) == 1)

	# Solving the model
	time = @timed optimize!(m)

	# Using QP solution to define the cluster
	alpha = round.(value.(m[:alpha]),digits=round_factor)
	not_rounded = value.(m[:alpha])

	# Set of support vector
	SV = [i for i in 1:N if alpha[i] > 0]

	# Set of boundary support vector
	BSV = [i for i in SV if (alpha[i] < round(1/(N*nu),digits=round_factor))]


	return alpha, SV, BSV
end

"""
	approximation(svc::SVC,K::Int)

	First phase of the approximation
	_ svc : The SVC object to approximate
	_ K the number of medoids
"""
function approximation(svc::SVC,K::Int)
	N::Int = size(svc.Data,2) # Get the number of sample points in the data set
	n::Int = size(svc.Data,1) # Dimension of the data
	# Select a subset of K SVs
	selected, groups = subset_selection(svc.Data,svc.Q,svc.SV,svc.BSV,K)
	# Update the weights alpha of tthe selected subset
	selected, beta, max_distance, LP_time = Min_Sum(svc.Data,svc.Q,svc.Alpha,selected,svc.SV,collect(1:N))
	# Build and return the SVC object
	approx = SVC(svc.Data,svc.nu)
	approx.SV = selected
	approx.Alpha = beta
	approx.BSV = svc.BSV
	approx.Theta = svc.Theta
	return approx, groups, max_distance, LP_time
end


"""
	subset_selection(X::Array{Float64,2}, Q::Array{Float64,2}, SV::Array{Int,1}, BSV::Array{Int,1},K::Int)

    Define the subset of point used to compute the new cluster (K-medoid clustering)
    _ X :  Initial set of coordinates
    _ Q : Weighting matrix of SVC algorithm
    _ SV : Set of support vectors
    _ BSV : Set of boundary support vectors
    _ K : Number of cluster
"""
function subset_selection(X::Array{Float64,2}, Q::Array{Float64,2}, SV::Array{Int,1}, BSV::Array{Int,1},K::Int)

    strict_sv = setdiff(SV,BSV) # Excluding BSV to define clusters
    SV_coord = X[:,strict_sv] # Loading coordinates
    # Define the distance matrix
    dist = Array{Float64,2}(undef,length(strict_sv),length(strict_sv))
    for i in 1:length(strict_sv)
        for j in 1:length(strict_sv)
            dist[i,j] = norm((SV_coord[:,i]-SV_coord[:,j]),2)
        end
    end

    R = kmedoids(dist, K,maxiter=10000) # Cluster object from Clustering.jl package
    a = assignments(R) # get the assignments of points to clusters
    c = counts(R) # get the cluster sizes
    M = med_ind = R.medoids # Index of medoids (i.e. center of a cluster)
    selected = [strict_sv[M[i]] for i in 1:K]
	groups = [[strict_sv[i] for i in 1:length(strict_sv) if a[i] == g] for g in 1:K]
	selected = vcat(selected,BSV) # Add the Boundary Support Vectors (reason : more precise approximation)

    return Array{Int,1}(selected), groups
end



"""
	Min_Sum(X::Array{Float64,2}, Q::Array{Float64,2},alpha_ini::Array{Float64,1}, selected::Array{Int,1}, SV::Array{Int,1},dist_set::Array{Int,1})

    Update contribution values (alpha) associated with a subset of sample points (SV) Min_SUm approach
    _ X :  Initial set of coordinates
    _ Q : Weighting matrix of SVC algorithm
    _ selected : subset of selected points
    _ SV : Set of support vectors
    _ dist_set : temporary parameter used to test hypothesis
"""
function Min_Sum(X::Array{Float64,2}, Q::Array{Float64,2},alpha_ini::Array{Float64,1}, selected::Array{Int,1}, SV::Array{Int,1},dist_set::Array{Int,1})
	# Initializing model and variables
    m = Model(CPLEX.Optimizer)
    @variable(m, D[i in dist_set] >= 0) # Maximum distance
    @variable(m, 0 <= w[i in selected]) # ws
	# Objective function and constraints
    @objective(m, Min, sum(D[i] for i in dist_set))
    @constraint(m, [i in dist_set], D[i] >= sum((alpha_ini[ii])*norm(Q*(X[:,i]-X[:,ii]),1) for ii in SV)
		- sum((w[ii])*norm(Q*(X[:,i]-X[:,ii]),1) for ii in selected))
    @constraint(m, [i in dist_set], D[i] >= sum((w[ii])*norm(Q*(X[:,i]-X[:,ii]),1) for ii in selected)
        - sum((alpha_ini[ii])*norm(Q*(X[:,i]-X[:,ii]),1) for ii in SV))
	set_optimizer_attribute(m, "CPXPARAM_Threads", 1)

	# Options for solving
    set_silent(m)
    solving_time = (@timed optimize!(m))[2] # Solving

	selected::Array{Int,1} = [i for i in selected if value.(m[:w])[i] > 0]
	weights = zeros(Float64,length(X[1,:]))
	for i in selected
		weights[i] = value.(m[:w])[i]
	end
    return selected, weights, value.(m[:D]), solving_time
end

"""
	get_dist(svc::SVC, approx::SVC, data_set::Matrix{Float64})

	Compute the different of theta^(i) between two SVC object on a given data-set
	_ svc : The initial cluster
	_ approx : The approximated cluster
	_ data_set : the data set... (Oh ! really ?)
"""
function get_dist(svc::SVC, approx::SVC, data_set::Matrix{Float64})
	return [abs(svc.value(data_set[:,i]) - approx.value(data_set[:,i])) for i in 1:length(data_set[1,:])]
end

"""
	phase_2(svc::SVC,app::SVC;time_limit::Int=3600)

	Second phase of the approximation
	_ svc : Initial SVC object
	_ app : Approximate SVC Object
	_ time_limit : The maximum CPLEX CPU-time
"""
function phase_2(svc::SVC,app::SVC;time_limit::Int=3600)
	# Compute theta^(i)
	dist = get_dist(svc,app,svc.Data)
	# Update the set of support vectors (minimize |SV|)
	selected, beta, max_distance, MIP_time = update_SV(svc,app,dist,time_limit=time_limit)
	# Update the weights alpha for the selected SV
	selected, beta, max_distance, LP_time = Min_Sum(svc.Data,svc.Q,svc.Alpha,selected,svc.SV,collect(1:size(svc.Data,2)))
	# Build and return the approximation (and other stuffs)
	ret = SVC(svc.Data,svc.nu)
	ret.SV = selected
	ret.Alpha = beta
	ret.BSV = svc.BSV
	ret.Theta = svc.Theta
	return ret, max_distance, MIP_time, LP_time
end

"""
	update_SV(svc::SVC,app::SVC,max_D::Vector{Float64};time_limit::Int=3600)

	Opitmization problem of phase 2. Minimize the number of SV
	_ svc : The initial SVC
	_ approx : The approximation obtained with phase 1
	_ max_D : The vector of theta^(i)
"""
function update_SV(svc::SVC,app::SVC,max_D::Vector{Float64};time_limit::Int=3600)
	X = svc.Data
	N = size(X,2)

	m = Model(CPLEX.Optimizer)
	@variable(m, D[i in 1:N] >= 0) # Maximum distance
    @variable(m, 0 <= beta[i in svc.SV] <= 1)
    @variable(m, y[i in svc.SV], Bin)

    for i in app.SV
        set_start_value(y[i], 1)
		set_start_value(beta[i], app.Alpha[i])
    end
	for i in setdiff(svc.SV,app.SV)
		set_start_value(y[i],0)
		set_start_value(beta[i], 0.0)
	end

	for i in 1:N
		set_start_value(D[i], max_D[i])
	end

	@objective(m, Min, sum(y[i] for i in svc.SV))

    @constraint(m, [i in 1:N], D[i] >= sum((svc.Alpha[ii])*norm(svc.Q*(X[:,i]-X[:,ii]),1) for ii in svc.SV)
        - sum((beta[ii])*norm(svc.Q*(X[:,i]-X[:,ii]),1) for ii in svc.SV))
    @constraint(m, [i in 1:N], D[i] >= sum((beta[ii])*norm(svc.Q*(X[:,i]-X[:,ii]),1) for ii in svc.SV)
        - sum((svc.Alpha[ii])*norm(svc.Q*(X[:,i]-X[:,ii]),1) for ii in svc.SV))
	@constraint(m, sum(D[i] for i in 1:N) <= sum(max_D))
	@constraint(m, [i in svc.SV], y[i] >= beta[i])

	# Options for solving
	set_optimizer_attribute(m, "CPXPARAM_Threads", 1)
	set_time_limit_sec(m,time_limit)
	# set_silent(m)
	solving_time = (@timed optimize!(m))[2] # Solving

	selected::Array{Int,1} = [i for i in svc.SV if value.(m[:beta])[i] > 0]
	weights = value.(m[:beta])
	return selected, weights, value.(m[:D]), solving_time
end

"""
    evaluate_svc(svc::SVC, solution; mode="Max")

	Solve the primal problem associated with a SVC
	_ svc : The SVC object
	_ solution : Solution to the robust problem
	_ mode : Optional Max or Min for the objective function
"""
function evaluate_svc(svc::SVC, solution; mode="Max")
    m = size(svc.Q,1)
    # Initialize model
    model = Model(CPLEX.Optimizer)

    # Create variables
    @variable(model, a[j in 1:m] >= 0)
    @variable(model, v[j in 1:m, i in svc.SV] >= 0)

    # Objective function
    if mode == "Max"
        @objective(model, Max, sum(a[j]*solution[j] for j in 1:m))
		println("Max")
    else
        @objective(model, Min, sum(a[j]*solution[j] for j in 1:m))
		println("Min")
    end

    # Constraint
    @constraint(model, sum(sum(svc.Alpha[i]*v[j,i] for i in svc.SV) for j in 1:m) <= svc.Theta)
    @constraint(model, [j in 1:m, i in svc.SV], sum(svc.Q[j,k]*(a[k]-svc.Data[k,i]) for k in 1:m) <= v[j,i])
    @constraint(model, [j in 1:m, i in svc.SV], sum(svc.Q[j,k]*(a[k]-svc.Data[k,i]) for k in 1:m) >= -v[j,i])

    # Solving
	set_optimizer_attribute(model, "CPXPARAM_Threads", 1)
    set_silent(model)
    optimize!(model)

    a = value.(model[:a])
    obj = objective_value(model)

    return obj, a
end


#===============================================================#
#																#
#           Not important things here just parsing files        #
#																#
#===============================================================#
function write(file::String,alpha::Array{Float64,1},SV::Array{Int,1},BSV::Array{Int,1},theta::Float64)
	f = open(file,"w")
		println(f,alpha)
		println(f,SV)
		println(f,BSV)
		println(f,theta)
	close(f)
end

"""
	Read the different components of a cluster from a .txt file

	* Params :
		_ file : the name of the file
"""
function read(file::String)
	# Open the file
	f = open(file,"r")
	# Parse the weights alpha
	splited = string.(Base.split(readline(f)))
	for i in 1:length(splited)
		if i == 1
			splited[i] = splited[i][2:end-1]
		else
			splited[i] = splited[i][1:end-1]
		end
	end
	alpha = parse.(Float64,splited)
	# Parse the Support Vector set
	splited = string.(Base.split(readline(f)))
	for i in 1:length(splited)
		if i == 1
			splited[i] = splited[i][2:end-1]
		else
			splited[i] = splited[i][1:end-1]
		end
	end
	SV = parse.(Int,splited)
	# Parse the Boundary Support Vector set
	splited = string.(Base.split(readline(f)))
	for i in 1:length(splited)
		if i == 1
			splited[i] = splited[i][2:end-1]
		else
			splited[i] = splited[i][1:end-1]
		end
	end
	BSV = parse.(Int,splited)
	# Read the bound theta
	theta = parse(Float64,readline(f))
	# Read the time

	time = readline(f)
	if time != ""
		time = parse(Float64,time)
	end
	# Close the file
	close(f)
	return alpha, SV, BSV, theta, time
end
