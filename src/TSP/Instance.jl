#=================================================================================================================#
# Author : Benoit Loger
#
# Description : Structure of instance and related methods
#
# Last update : 21/10/2023 (dd/mm/yyyy)
#
#=================================================================================================================#
mutable struct Instance
	nb_nodes::Int # Number of city
	X::Array{Float64,1} # x coordinates of the cities
	Y::Array{Float64,1} # y coordinates of the cities
	Pert_tab::Array{Tuple{Int,Int},1} # Set of edges with uncertain travel time
	Data::Array{Float64,2} # historical travel time data

	function Instance(name::String;path::String="Instances/")
		inst_path = string(path,name)

		# Initialisation de la structure par champs
		df_coord = DataFrame(CSV.File(string(inst_path,"coord.csv")))
		df_data = DataFrame(CSV.File(string(inst_path,"data.csv")))
		pert_tab = read_pert_tab(string(inst_path,"pert.txt"))
		# Getting values
		x = df_coord[!,"x1"]
		y = df_coord[!,"x2"]
		nb_nodes = length(x)

		return new(nb_nodes,x,y,pert_tab,Array{Float64,2}(df_data))
	end

end

"""
	read_pert_tab(file::String)

	Read the list of uncertain edges
	_ file : the file where this list is stored
"""
function read_pert_tab(file::String)
	pert_tab = []
	f = open(file,"r")
		for l in eachline(f)
			tab = parse.(Int,Base.split(l))
			push!(pert_tab,(tab[1],tab[2]))
		end
	close(f)
	return pert_tab
end

"""
	get_dist(inst::Instance,i::Int,j::Int)

	Compute euclidean distance
	_ inst : Instance structure
	_ i,j : Nodes
"""
function get_dist(inst::Instance,i::Int,j::Int)
	return sqrt((inst.X[i]-inst.X[j])^2+(inst.Y[i]-inst.Y[j])^2)
end
