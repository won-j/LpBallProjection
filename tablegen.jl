using LinearAlgebra

"""Checks the KKT conditions for projecting y onto the ball 
||x||_p <= r."""
function output(y::Vector{T}, x::Vector{T}, p::T, r::T, secs::T,
  iters::Int, method::Int, df; mu::T=convert(T, NaN)) where T <: Real
  n = length(y)
  if isnan(mu)
    ###mu = dot(y - x, x) / r^p
    #mu = dot(abs.(y) - abs.(x), abs.(x)) / r^2
    ##mu = dot(abs.(y) - abs.(x), abs.(x)) / r^2 / norm(x / r, p)^p # more precise?
	#mu = dot(abs.(y) - abs.(x), abs.(x)) / norm(x, p)^p # independent of `r`
	mu = dot(abs.(y / r) - abs.(x / r), abs.(x / r)) / norm(x / r, p)^p  # in normalized primal scale
  end
  #mu2 = mu / norm(x / r, p)^p 	# more precise?
  mu2 = mu
  cutoff = 1e-8 * maximum(abs.(x))
  KKT  = zero(T)  # Lagrange dual optimality condition
  KKT2 = zero(T)  

  xx = sign.(y) .* (x / r) # convert to nonnegative entries and r=1
  yy = sign.(y) .* (y / r) # convert to nonnegative entries and r=1
  pval = 0.5 * norm(xx - yy)^2   # primal objective
  dval = (mu2 < zero(T)) ? NaN : (-mu2 / p)  # dual objective, to be updated
  #pval = 0.5 * norm(x - y)^2 	# primal objective
  #dval = (mu < zero(T)) ? NaN : (- mu / p * r^p)

  q = p / (p - 1)
  # Fenchel dual: z = y - x
  nrm =  norm(y - x, q) 	# ||z||_q
  if !(nrm > zero(T))
  	KKT2 = maximum([norm(y, p) - r, zero(T)])
  end
  for i = 1:length(y)
    if abs(x[i] / r)^(p - 1)  < 1e20 * abs(x[i] / r)
      d = abs(x[i] / r - y[i] / r + sign(y[i]) * ((abs(x[i]) < 1e-12) ? abs(y[i]) / r : mu * abs(x[i] / r)^(p - 1)))
      KKT += d
    end
	# Fenchel optimality: z / r - y / r + ||z||_q^(1 - q) * z^(q - 1) = 0
	# where z^(q - 1) = [ sign(z[i]) * abs(z[i])^(q - 1) for i=1:n]
	if nrm > zero(T)
	  z = y[i] - x[i]
	  KKT2 += abs( z / r - y[i] / r + sign(y[i]) * (abs(z) < 1e-12 ? abs(y[i]) / r : (abs(z) / nrm)^(q - 1)) )
	end

	## dual objective
##println("yy[i] = ", yy[i], ", mu2 = ", mu2, ", mu = ", mu)
	if mu2 >= zero(T)
		xmu, _ = LpProx(yy[i], max(zero(T), mu2), p, 1e-12)
		dval += 0.5 * one(T) * (yy[i] - xmu)^2  + mu2 / p * abs(xmu)^p
	end
	#if mu >= zero(T)
	#	xmu, _ = LpProx(yy[i], mu, p, 1e-12)
	#	dval += 0.5 * one(T) * (y[i] - xmu)^2  + mu / p * abs(xmu)^p
	#end
  end
  gap = pval - dval # duality gap
  constr = norm(x, p) / r - one(T)  # constraint satisfaction
  #success = (abs(constr) < 1e-5)
  success = !isnan(gap)
  #push!(df, [method, p, iters, secs, KKT, KKT2, norm(y - x), norm(x, p) / r - one(T), support])
  push!(df, [method, p, iters, secs, KKT, gap, norm(y - x), constr, success])
end

