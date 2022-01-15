#
# Compare dual bisection and IRBP
#
include("projectionmaps.jl")
include("tablegen.jl")
using LinearAlgebra, DataFrames, Statistics, CSV
using Random
using PyCall

Random.seed!(1234)
df = DataFrame(Method=Int[], Power=Float64[], Iters=Float64[], 
  Secs=Float64[], KKT=Float64[], KKT2=Float64[], Obj=Float64[], Radius=Float64[], Support=Int[])
pv = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]; # power
(n, reps, tol) = (1000000, 100, 1e-12); # dimension, repetitions, tolerance
nummethods = 5
numcols = 9
Means = zeros(nummethods, length(pv), numcols);
# compute mean excluding NaNs
nanmean(x) = mean(filter(!isnan, x))
nanmean(x, y) = mapslices(nanmean, x, dims=y)
# setup for python implementation of IRBP
pushfirst!(pyimport("sys")."path", "")  # python search path
irbp = pyimport("run_irbp_lib")
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
    secs = @elapsed (x, mu, history) = BisectionProjection(y, p, r, proxtol=tol); # bisection
    output(y, x, p, r, secs, history.iters, 7, df, mu = mu);
	secs = @elapsed (x, mu, history) = IRBP1(y, p, r, maxiter = 1000); # iterative reweighted L1 ball projection
    output(y, x, p, r, secs, history.iters, 10, df) #, mu = mu);
	secs = @elapsed (x, mu, history) = IRBP2(y, p, r, maxiter = 1000); # iterative reweighted L1 ball projection
    output(y, x, p, r, secs, history.iters, 11, df) #, mu = mu);
	secs = @elapsed x_py, mu_py, iters_py = irbp.get_sol(y / r, p, 1.0); # IRBP reference implementation
	x = x_py * r
    output(y, x, p, r, secs, iters_py, 12, df);
  end
  sort!(df, (:Method))
  X = Matrix(df)
  for j = 1:nummethods # loop over methods
    a = (j-1)*reps + 1
    b = (j-1)*reps + reps
    Means[j, i, :] = nanmean(replace(X[a:b, :], Inf=>NaN), 1)
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
rename!(df, :x6 => :duality_gap)
rename!(df, :x7 => :obj)
rename!(df, :x8 => :norm_error)
rename!(df, :x9 => :success_rate)
df[!,:method] = convert.(Int,df[!,:method])
# add LatexPrint
using LatexPrint
AA = round.(A, sigdigits=4)
C = convert(Matrix{Any}, AA)
C[:, 1] = Int.(C[:, 1])
B = lap(C)
print(df)
println()
