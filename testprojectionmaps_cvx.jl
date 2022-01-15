#
# Compares Newton's method and block descent on the dual
#
include("projectionmaps.jl")
include("tablegen.jl")
using LinearAlgebra, DataFrames, Statistics, CSV
using Random
Random.seed!(1234)
df = DataFrame(Method=Int[], Power=Float64[], Iters=Float64[], 
  Secs=Float64[], KKT=Float64[], KKT2=Float64[], Obj=Float64[], Radius=Float64[], Support=Int[])
pv = [1.01, 1.05, 1.1, 1.5, 4.0, 10.0, 99.0, 100.0]; # power
(n, reps, tol) = (1000000, 100, 1e-12); # dimension, repetitions, tolerance
nummethods = 4
numcols = 9
Means = zeros(nummethods, length(pv), numcols);
for i = 1:length(pv)
println(stderr, i," of ",length(pv))
  p = pv[i]
  for rep = 1:reps
	rep % 10 == 0 && println(stderr, "rep = ", rep)
    y = randn(n); # external point
    r = rand() * norm(y, p)
    if norm(y, p) == Inf
      println(stderr, "Infinite norm")
    end
    secs = @elapsed (x, iters) = NearestExactProjection(y, p, r); # nearest exact
    output(y, x, p, r, secs, iters, 1, df);
    secs = @elapsed (x, mu, history) = NewtonProjection(y, p, r, proxtol=tol); # Newton's method
    output(y, x, p, r, secs, history.iters, 6, df, mu = mu);
    secs = @elapsed (x, mu, history) = BisectionProjection(y, p, r, proxtol=tol); # bisection
    output(y, x, p, r, secs, history.iters, 7, df, mu = mu);
    secs = @elapsed (x, info) = ProjectedNewtonPTV(y, p, r); # projected Newton (direct call to proxTV)
    output(y, x, p, r, secs, convert(Integer, info[1]), 8, df);
  end
  sort!(df, (:Method))
  X = Matrix(df)
  for j = 1:nummethods # loop over methods
    a = (j-1)*reps + 1
    b = (j-1)*reps + reps
    Means[j, i, :] = mean(X[a:b, :], dims=[1])
  end
  #deleterows!(df, 1:(nummethods *reps))
  delete!(df, 1:(nummethods *reps))
end
# CSV.write("TestProjection.dat", Stats)
# df = CSV.read("TestProjection.dat")
A = reshape(Means, nummethods * length(pv), numcols)
#df = convert(DataFrame, A)	# deprecated
df = DataFrame(Tables.table(A, header=Symbol.(:x, axes(A, 2))))
rename!(df, :x1 => :method)
rename!(df, :x2 => :power)
rename!(df, :x3 => :iters)
rename!(df, :x4 => :secs)
rename!(df, :x5 => :KKT)
rename!(df, :x6 => :KKT2)
rename!(df, :x7 => :obj)
rename!(df, :x8 => :norm_error)
rename!(df, :x9 => :support)
df[!,:method] = convert.(Int,df[!,:method])
# add LatexPrint
using LatexPrint
AA = round.(A, sigdigits=4)
C = convert(Matrix{Any}, AA)
C[:, 1] = Int.(C[:, 1])
B = lap(C)
print(df)
println()
