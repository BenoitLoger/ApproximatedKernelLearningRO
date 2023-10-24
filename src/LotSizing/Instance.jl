
#=================================================================================================================#
# Author : Benoit Loger
#
# Description : Structure of instance and related methods
#
# Last update : 21/10/2023 (dd/mm/yyyy)
#
#=================================================================================================================#
mutable struct Instance
	# Attributes
	T::Int # Planning horizon
	C::Float64 # Unit ordering cost
	P::Float64 # Unit shortage cost
	H::Float64 # Unit holding cost
	K::Float64 # Fixed ordering cost
	W::Matrix{Int} # Set of historical demand scenarios

	write::Function
	# Simple constructor for developement
	Instance() = new()

	Instance(t::Int,
			c::Float64, # Unit ordering cost
			p::Float64, # Unit shortage cost
			h::Float64, # Unit holding cost
			k::Float64, # Fixed ordering cost
			w::Matrix{Int} # Set of historical demand scenarios
			) = new(t,c,p,h,k,w)


	function Instance(name::String;path="Instances/")
		W = Matrix{Int}(round.(Matrix{Float64}(DataFrame(CSV.File(string(path,name,"/demands.csv"))))))
		T = length(W[:,1])
		f = open(string(path,name,"/costs.txt"),"r")
			c = parse.(Float64,Base.split(readline(f)))
		close(f)
		return Instance(T,c[1],c[2],c[3],c[4],W)
	end
end
