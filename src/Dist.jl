#=================================================================================================================#
# Author : Benoit Loger
#
# Description : Contain methods used to save information on the distribution used
#				in a given problem instance. Avoid to sistematically generate and
#				savve *LARGE* test data sets.
#
# Last update : 11/06/2020 (dd/mm/yyyy)
#
#=================================================================================================================#
# Abstract structure of Dist(ribution)
abstract type Dist end

# Structure of multivariate Normal distribution
mutable struct normal <: Dist
	Type::String # Type of Dist (Normal)
	Mean::Array{Float64,1} # Mean value of each marginal dist
	Std::Array{Float64,1} # Standard deviation of each marginal
	Cor::Array{Float64,2} # Correlation matrix
	Cov::Array{Float64,2} # Covariance matrix

	# Methods (fake Object-Oiented Programming)
	generate::Function # Generate rand samples
	write::Function # Save the given distribution in a file

	"""
		Constructor of normal distribution objects
	"""
	function normal(Mean::Array{Float64,1}, Std::Array{Float64,1}, Cor::Array{Float64,2}, Cov::Array{Float64,2})
		# Initialise
	 	this = new("Normal",Mean,Std,Cor,Cov)

		# Implement generate function
		this.generate = function(N::Int)
			return rand(MvNormal(this.Mean,this.Cov),N)
		end

		# Implement write function
		this.write = function(file::String)
			m = length(this.Mean)
			f = open(file,"w")
				println(f,this.Type)
				println(f,this.Mean)
				println(f,this.Std)
				for i in 1:m
					println(f,Cor[:,i])
				end
				for i in 1:m
					println(f,Cov[:,i])
				end
			close(f)
		end

		return this
	end

	"""
		Constructor of normal distribution from a file
	"""
	function normal(file::String;path::String="")
		# Initialise
	 	this = new("Normal")
		this.Mean, this.Std, this.Cor, this.Cov = read_normal(string(path,file))

		# Implement generate function
		this.generate = function(N::Int)
			return rand(MvNormal(this.Mean,this.Cov),N)
		end

		return this
	end
end

# Structure of multivariate Gamma Dist
mutable struct gamma <: Dist
	Type::String # Type of Dist (Gamma)
	K::Array{Float64,1} # Shape parameter of each marginal distribution
	Theta::Array{Float64,1} # Scale parameter of each marginal
	Cor::Array{Float64,2} # Corelation matrix

	# Methods (fake Object-Oiented Programming)
	generate::Function # Generate rand samples
	write::Function # Save the given distribution in a file

	"""
		Constructor of gamma distribution objects
	"""
	function gamma(K::Array{Float64,1}, Theta::Array{Float64,1}, Cor::Array{Float64,2})
		# Initialise
	 	this = new("Gamma",K,Theta,Cor)

		# Implement generate function
		this.generate = function(N::Int)
			return multivariate_gamma(this.K,this.Theta,N,Cor=this.Cor)[1]
		end

		# Implement write function
		this.write = function(file::String)
			m = length(this.K)
			f = open(file,"w")
				println(f,this.Type)
				println(f,this.K)
				println(f,this.Theta)
				for i in 1:m
					println(f,Cor[:,i])
				end
			close(f)
		end
		return this
	end

	"""
		Constructor of gamma distribution objects from a file
	"""
	function gamma(file::String;path::String="")
		# Initialise
	 	this = new("Gamma")
		this.K, this.Theta, this.Cor = read_gamma(string(path,file))

		# Implement generate function
		this.generate = function(N::Int)
			return multivariate_gamma(this.K,this.Theta,N,Cor=this.Cor)[1]
		end

		return this
	end


end

#===============================================================#
#																#
#           Not important things here just parsing files        #
#																#
#===============================================================#
function read_normal(file::String)
	f = open(file,"r")
	readline(f)
	splited = string.(Base.split(readline(f)))
	for i in 1:length(splited)
		if i == 1
			splited[i] = splited[i][2:end-1]
		else
			splited[i] = splited[i][1:end-1]
		end
	end
	Mean = parse.(Float64,splited)
	n = length(Mean)
	# println(Mean)

	splited = string.(Base.split(readline(f)))
	for i in 1:length(splited)
		if i == 1
			splited[i] = splited[i][2:end-1]
		else
			splited[i] = splited[i][1:end-1]
		end
	end
	Std = parse.(Float64,splited)
	# println(Std)

	Cor = Array{Float64,2}(undef,n,n)
	for i in 1:n
		splited = string.(Base.split(readline(f)))
		for i in 1:length(splited)
			if i == 1
				splited[i] = splited[i][2:end-1]
			else
				splited[i] = splited[i][1:end-1]
			end
		end
		Cor[i,:] = parse.(Float64,splited)
	end
	# println(Cor)

	Cov = Array{Float64,2}(undef,n,n)
	for i in 1:n
		splited = string.(Base.split(readline(f)))
		for i in 1:length(splited)
			if i == 1
				splited[i] = splited[i][2:end-1]
			else
				splited[i] = splited[i][1:end-1]
			end
		end
		Cov[i,:] = parse.(Float64,splited)
	end
	# println(Cov)
	close(f)
	return Mean,Std, Cor, Cov
end

function read_gamma(file::String)
	f = open(file,"r")
	readline(f)
	splited = string.(Base.split(readline(f)))
	for i in 1:length(splited)
		if i == 1
			splited[i] = splited[i][2:end-1]
		else
			splited[i] = splited[i][1:end-1]
		end
	end
	k = parse.(Float64,splited)
	n = length(k)
	# println(k)

	splited = string.(Base.split(readline(f)))
	for i in 1:length(splited)
		if i == 1
			splited[i] = splited[i][2:end-1]
		else
			splited[i] = splited[i][1:end-1]
		end
	end
	theta = parse.(Float64,splited)
	# println(theta)
	Cor = Array{Float64,2}(undef,n,n)
	for i in 1:n
		splited = string.(Base.split(readline(f)))
		for i in 1:length(splited)
			if i == 1
				splited[i] = splited[i][2:end-1]
			else
				splited[i] = splited[i][1:end-1]
			end
		end
		Cor[i,:] = parse.(Float64,splited)
	end
	close(f)
	return k, theta, Cor
end
