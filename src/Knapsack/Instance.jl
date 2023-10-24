#=================================================================================================================#
# Author : Benoit Loger
#
# Description : Structure of instance and related methods
#
# Last update : 21/10/2023 (dd/mm/yyyy)
#
#=================================================================================================================#
# Instance of Knapsack Problem
mutable struct Instance
	c::Array{Float64,1} # Cost vector
	dist::Dist # Dist(ribution)
	w::Array{Float64,2} # Set of historical weights
	b::Int # Maximum value of

	write::Function

	# Constructor
	function Instance(c::Array{Float64,1},dist::Dist,w::Array{Float64,2},b::Int)
		# Initialisation de la structure par champs
		this = new(c,dist,w,b)
		this.write = function(name::String;path::String="")
			# Create folder
			location = pwd()
			cd(path)
			mkdir(name)
			cd(name)
			dist.write("dist.txt") # dist file
			write("weights.csv",this.w) # weights file
			f = open("costs.txt","w") # cost file
				println(f,this.c)
				println(f,b)
			close(f)
			cd(location)
		end
		return this
	end

	"""
		function load_sol(solution_path::String,solution_file::String)

		Load the solutions stored in a given file
		_ solution_path : Path to the folder
		_ solution file : File name
	"""
	function load_sol(solution_path::String,solution_file::String)
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

	"""
		function Instance(name::String,ind::Int;N=1000,path::String="")

		Load an existing instance
		_ name : name of the instance (e.g. gamma_10)
		_ ind : number of the instance (10 per name)
		_ N : extension for test with different N
		_ path : path to instance folder
	"""
	function Instance(name::String,ind::Int;N=1000,path::String="")
		inst_path = string(path,name,"")
		c,b = read_info(string(inst_path,"costs/costs_",ind,".txt"))
		if N == 1000
			w = Array{Float64,2}(DataFrame(CSV.File(string(inst_path,"weights/weights_$ind.csv"))))
		end
		f = open(string(inst_path,"dist.txt"),"r")
			d = readline(f)
		close(f)
		if d == "Normal"
			dist = normal(string(inst_path,"dist.txt"))
			return new(c,dist,w,b)
		elseif d == "Gamma"
			dist = gamma(string(inst_path,"dist.txt"))
			return new(c,dist,w,b)
		end
	end

end


#===============================================================#
#																#
#           Not important things here just parsing files        #
#																#
#===============================================================#
function write(file::String,w::Array{Float64,2})
	df = DataFrame(w,:auto)
	CSV.write(file,df)
end

function read_info(file)
	f = open(file,"r")
		splited = string.(Base.split(readline(f)))
		for i in 1:length(splited)
			if i == 1
				splited[i] = splited[i][2:end-1]
			else
				splited[i] = splited[i][1:end-1]
			end
		end
		c = parse.(Float64,splited)
		b = parse(Int,readline(f))
	close(f)
	return c, b
end
