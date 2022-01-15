## compressed sensing via Lp norm ball projection
include("projectionmaps.jl")

# random seed
using Random, LinearAlgebra, DataFrames, CSV, Dates
Random.seed!(280)

# Problem parameters
# p = 0.5
#p = 0.0
p = 1.0
mgrid = 5
sgrid = 2
numrep = 5
df = DataFrame(p = Float64[], n = Int[], m = Int[], s = Int[], success = Bool[])

# Algorithm paramters
maxiter = 500
succthres = 1e-3  # threshold for declaring success

# Size of signal
n = 1000
# Sparsity (# nonzeros) in the signal
for k = 1:sgrid
	s = n ÷ sgrid * k
	#s = 20

	# Generate sparse signal
	x0 = zeros(n)
	x0[rand(1:n, s)] = randn(s)
	r  = norm(x0, p) # projection radius

	for i = 1:mgrid
		# Number of measurements 
		m = 6 * n ÷ mgrid * i	
		#m = 250

		for rep = 1:numrep
			println("n = ", n, ", m = ", m, ", s = ", s )
			# Generate the random sensing matrix
			A = randn(m, n) 
			# Measurements (noiseless)
			y = A * x0

			# PGD
			gam = 1 / m		# step size
			x = zeros(size(x0)) # initial point
			xold = x
			for k=1:maxiter
				#println("k = ", k)
				#global x, xold, r
				gstep = x + gam * (A' * (y - A * x))	# gradient step
				#(x, mu, history) = BisectionProjection(gstep, p, r)
				x = sign.(gstep) .* SimplexProjection(abs.(gstep), r) # L1

  				#x = copy(gstep) # L0
  				#perm = partialsortperm(gstep, by = abs, 1:(n - Int(r)))
  				#for i = 1:(n - Int(r))
    			#	x[perm[i]] = zero(Float64)
  				#end

				if norm(x - xold) < 1e-5
					break
				end
				xold = x
			end
			reconerr = norm(x - x0) / norm(x0)
			#println("error = ", reconerr)
			#if reconerr < 1e-3
			#	println("success!")
			#end
			push!(df, (p, n, m, s, reconerr < succthres))
		end
	end
end

print(df)
println()
ts = now()
CSV.write("output_p$p" * 
		  "-" * string(Dates.today()) * 
		  "-" * string(Dates.hour(ts)) * 
		  "-" * string(Dates.minute(ts)) *
		  "-" * string(Dates.second(ts)) * 
		  ".csv", df)

