#=================================================================================================================#
# Author : Benoit Loger
#
# Description : Contain methods used to generate random correlated samples
#
# Last update : 11/06/2020 (dd/mm/yyyy)
#
#=================================================================================================================#
"""
    random_cor(d::Int,k::Int;Min::Float64=-100.0,Max::Float64=100.0)

    Generate rondom correlation matrix of dimension d*d
    _ d : Dimension of the matrix
    _ k : Any positive integer (higher k lead to lower average absolut correlation)
    _ Min/Max :
"""
function random_cor(d::Int,k::Int;Min::Float64=-100.0,Max::Float64=100.0)
    W = rand((Min:0.001:Max),d,k)
    r = diagm(rand(d))
    S = W*W' + r
    X = rand(MvNormal([0.0 for i in 1:d],S),10000)
    return cor(X,dims=2)
end

"""
    rand_from_marginal(marginals::Vector{Distribution}, cor_matrix::Matrix, nb_sample::Int)

    Generate random samples from marginal distributions (see package Distributions.jl) and a correlation matrix
    _ marginals : Vector of marginal distributions
    _ cor_matrix : Correlation matrix
    _ nb_sample : Number of sample to generate
"""
function rand_from_marginal(marginals::Vector{Distribution}, cor_matrix::Matrix, nb_sample::Int)
    m = length(marginals)

    # Initial multivariate normal distribution with mean 0.0 standard deviation 1.0 and the given correlation
    Initial_dist = MvNormal(zeros(Float64,m),cor_matrix)
    Initial_set = rand(Initial_dist,nb_sample)

    # Cumulated distribution function
    Initial_cdf = cdf.(Normal(),Initial_set)

    X = Matrix{Float64}(undef,m,nb_sample)
    for i in 1:m
        X[i,:] = quantile.(marginals[i], Initial_cdf[i,:])
    end

    return X
end

"""
    multivariate_normal(Mean::Vector{Float64},Std::Vector{Float64},N::Int;Cor=[],k::Int=5)

    Generate random sample with multivariate normal distribution with given mean and standard deviation
    _ Mean : Mean values
    _ Std : Standard deviations
    _ N : Number of samples
    _ Cor : Optional correlation matrix
    _ k : k parameter of random_cor() method
"""
function multivariate_normal(Mean::Vector{Float64},Std::Vector{Float64},N::Int;Cor=[],k::Int=5)
	m = length(Mean)
	marginals = []
	for i in 1:m
		push!(marginals,Normal(Mean[i],Std[i]))
	end
    if Cor == []
		Cor = random_cor(m,k)
	end
	X = rand_from_marginal(marginals,Cor,N)
	return X, Cor, cov(X,dims=2)
end

"""
    multivariate_gamm(shape::Vector{Float64},scale::Vector{Float64},N::Int;Cor=[],k::Int=5)

    Generate random sample with multivariate gamma distribution with given mean and standard deviation
    _ Mean : Shape parameter values
    _ Std : Scale parameter values
    _ N : Number of samples
    _ Cor : Optional correlation matrix
    _ k : k parameter of random_cor() method
"""
function multivariate_gamma(shape::Vector{Float64},scale::Vector{Float64},N::Int;Cor=[],k::Int=5)
	m = length(shape)
	marginals = Distribution[]
	for i in 1:m
		push!(marginals,Gamma(shape[i],scale[i]))
	end
	if Cor == []
		Cor = random_cor(m,k)
	end
	X = rand_from_marginal(marginals,Cor,N)
	return X, Cor, cov(X,dims=2)
end
