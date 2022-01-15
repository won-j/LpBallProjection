using IterativeSolvers

"""
	Project y onto the ball ||x||_p <= r via Fenchel dual by Projected Newton.
	This code directly calls
	https://github.com/albarji/proxTV/blob/master/src/LPopt.cpp
"""
const mylib = joinpath(@__DIR__, "libproxtv.dylib")
function ProjectedNewtonPTV(y::Vector{Float64}, p::Float64, r::Float64)
	x = copy(y)
	info = Vector{Cdouble}(undef, 3)
	ret = ccall((:LPp_project_wrapper, mylib), Cint, 
		  (Ptr{Cdouble}, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Cint, Cdouble), 
		  y, r, x, info, length(y), p
		 ) 
	x, info, ret
end

"""
Projects y onto the ball ||x||_p <= r by Newton's method.
"""
function NewtonProjection(y::Vector{T}, p::T, r::T; 
						  tol::T=convert(T, 1e-12),
						  proxtol::T=convert(T, 1e-12),
						  overflowtol::T=convert(T, 1e5),
						  maxiter::Integer=1000,
					   	  log::Bool = false
						 ) where T <: Real
  n, q, pnorm = length(y), 1 / (1 - 1 / p), norm(y, p)
  if pnorm <= r
    return (y, zero(T), 0)
  end
  if p == 2 * one(T)
  	# mu = norm(y, 2) /r - 1
    return ((r / pnorm) * y, norm(y, 2) / r - one(T), 0)
  end
  z = sign.(y) .* (y / r) # convert to nonnegative entries and r=1
  x = z / norm(z, p) # initial point, norm(x, p) == 1
  mu = dot(z - x, x) # initial multiplier. use 0 = x[i] - z[i] + x[i]^(p - 1) and sum([x[i]^p for i=1:d]) == one(T)
  (old_obj, iters) = (Inf, 0) 
  history = ConvergenceHistory(partial = !log)
  history[:tol] = tol
  IterativeSolvers.reserve!(T,       history, :objval,   maxiter)
  IterativeSolvers.reserve!(T,       history, :deriv1,   maxiter)
  IterativeSolvers.reserve!(T,       history, :mu,       maxiter)
  IterativeSolvers.reserve!(Float64, history, :itertime, maxiter)
  #IterativeSolvers.nextiter!(history)

  for iter = 1:maxiter # Newton iterations
    #iters = iter
	tic = time()
	# prox
    @inbounds for i = 1:n 
      (x[i], j) = LpProx(z[i], mu, p, proxtol)
	end
	xmax = maximum(x)
	xmaxp = xmax^p
	# objective, 1st_deriv, 2nd_deriv
    g0 = (xmaxp > overflowtol) ? - mu / p / xmaxp : - mu / p
    g2 = zero(T)
	g1 = (xmaxp > overflowtol) ? -1 / xmaxp / p : -1 / p
	if p < 2 * one(T)
		@inbounds for i = 1:n
			xp = x[i]^p
    		g0 += (z[i] - x[i])^2 / 2 + (mu / p) * xp
    		g1 += xp / p
    		d = x[i]^(2 - p) + mu * (p - 1)
    		g2 -= xp / d
		end
	elseif xmaxp > overflowtol	# scale g0, g1 and g2
		@inbounds for i = 1:n
    		g0 += (z[i] / xmax - x[i] / xmax)^2 / 2 + (mu / p) * (x[i] / xmax)^p
    		g1 += (x[i] / xmax)^p / p
    		d = (x[i] / xmax)^2 + mu * (p - 1) * (x[i] / xmax)^p
    		g2 -= (x[i] / xmax)^(2p) / d
		end
	else
		@inbounds for i = 1:n
			xp = x[i]^p
    		g0 += (z[i] - x[i])^2 / 2 + (mu / p) * xp
    		g1 += xp / p
    		d = x[i]^2 + mu * (p - 1) * xp
    		g2 -= xp^2 / d
		end
	end
	push!(history, :objval,   g0)
	push!(history, :deriv1,   g1)
	push!(history, :mu,       mu)
	# convergence check
    if abs(old_obj - g0) < tol * (one(T) + old_obj)
#println("Backtracking Newton: mu = ", mu, ", g1 = ", g1, ", g2=", g2)
	  toc = time()
	  push!(history, :itertime, toc - tic)
	  IterativeSolvers.setconv(history, true)
      break
    end
	## update mu
    old_obj = g0
    d = max(- g1 / g2, - mu) # Newton increment
    for step = 0:10 # step halving
      mu = mu + d
      g0 = - mu / p
	  if (p >= 2 * one(T)) && (xmaxp > overflowtol)	# scale g0
      	@inbounds for i = 1:n
        	(x[i], j) = LpProx(z[i], mu, p, proxtol)
        	g0 += (z[i] / xmax - x[i] / xmax)^2 / 2 + (mu / p) * (x[i] / xmax)^p
      	end
	  else
      	@inbounds for i = 1:n
        	(x[i], j) = LpProx(z[i], mu, p, proxtol)
        	g0 += (z[i] - x[i])^2 / 2 + (mu / p) * x[i]^p
      	end
	  end
      if g0 > old_obj
        break
      else
        mu -= d
        d /= 2
      end
    end
    #if old_obj > g0
    #  println(stderr, "ascent failure")
    #end
	toc = time()
	push!(history, :itertime, toc - tic)
	IterativeSolvers.nextiter!(history)
  end
  @inbounds for i = 1:n 
     x[i], j = LpProx(z[i], mu, p, proxtol)
  end
  # back to original scale
  x = sign.(y) .* (r * x)
  #mu = exp(Base.log(mu) - (p - 2) * Base.log(r) )  # mu = mu * r / r^(p - 1)
  #return (x, iters)
  log && IterativeSolvers.shrink!(history)
  x, mu, history
end


"""
Projects y onto the ball ||x||_p <= r by bisection for p < 1.
"""
function BisectionProjection(y::Vector{T}, p::T, r::T; 
							 tol::T=convert(T, 1e-12), 
							 proxtol::T=convert(T, 1e-12), 
							 maxiter::Integer=1000,
					   		 log::Bool = false
							) where T <: Real
  n, q, pnorm = length(y), 1 / (1 - 1 / p), norm(y, p)
  if pnorm <= r
    return (y, zero(T), 0)
  end
  if p == 2 * one(T)
  	# mu = norm(y, 2) /r - 1
    return ((r / pnorm) * y, norm(y, 2) / r - one(T), 0)
  end
  z = sign.(y) .* (y / r) # convert to nonnegative entries and radius 1
  #(a, b) = (zero(T), one(T)) # bracketing interval for multiplier
  #(ra, rb) = (pnorm, LpRadius(z, b, p, proxtol))
  #while (ra - one(T)) * (rb - one(T)) >= zero(T)
  #  b = 2 * b
  #  rb = LpRadius(z, b, p, tol)
  #end
  q = p / max(0, p - 1)
  (a, b) = (zero(T), norm(z, q)) # bracketing interval for multiplier
  (ra, rb) = (pnorm, LpRadius(z, b, p, proxtol))
  history = ConvergenceHistory(partial = !log)
  history[:tol] = tol
  IterativeSolvers.reserve!(T, history, :deriv1, maxiter)
  IterativeSolvers.reserve!(T, history, :mu, maxiter)
  IterativeSolvers.reserve!(Float64, history, :itertime, maxiter)
  IterativeSolvers.nextiter!(history)

  #numiters = maxiter
  for iteration = 1:maxiter # bisection loop on radius
	tic = time()
    m = (a + b) / 2	# mu
    rm = LpRadius(z, m, p, tol)  # 1st derivative
	push!(history, :deriv1,   rm)
	push!(history, :mu,       m)
	# convergence check
    if ((abs(ra - rb) < 1e-7 * (one(T) + ra)) || ( (b - a) / b < eps(T))) && ((b - a) < tol)
      #numiters = iteration
	  toc = time()
	  push!(history, :itertime, toc - tic)
	  IterativeSolvers.setconv(history, true)
      break
    end
	## update mu
    if (ra - one(T)) * (rm - one(T)) < zero(T) 
      (b, rb) = (m, rm)
    else
      (a, ra) = (m, rm)
    end
	toc = time()
	push!(history, :itertime, toc - tic)
	IterativeSolvers.nextiter!(history)
  end # end for
  # if you don't care multiple jumps
  #mu = (a + b) / 2
  #x = zeros(T, n)
  #for i = 1:n
  #  (x[i], iters) = LpProx(z[i], mu, p, proxtol)
  #end
  if ( (ra - rb) / (one(T) + ra) > 1e-7 ) # handle discontinuity when p < 1
	# ra > 1 and rb < 1
  	xleft  = [LpProx(z[i], a, p, proxtol)[1] for i=1:n] 
  	xright = [LpProx(z[i], b, p, proxtol)[1] for i=1:n] 
	xdiff  = xleft - xright  # nonnegative
	# xright[i] must be zero at the jump
	# detect this jump relative to xleft[i]:
	jmpidx = findall(x -> x > tol, xdiff./xleft)  # xleft > 0
	if length(jmpidx) > 1  # multiple coordinates contribute
#println("Multiple jumps detected!")
		xjump = xdiff[jmpidx] # jump sizes, idealy all equal
		sortedidx = sortperm(xjump)  # ascending order
		rbp = rb^p
		inc = zeros(T, length(sortedidx))
		j = 1
		inc[j] = xjump[sortedidx[1]]^p
		while rbp + inc[j] < 1.0
			j += 2
			inc[j] = inc[j-1] + xjump[sortedidx[j]]^p
		end
		# compare the difference from 1.0
		if j > 1 && 1.0 - rbp - inc[j-1] < rbp + inc[j] - 1.0
			j -= 1
		end
		x   = xright
		x[jmpidx[sortedidx[1:j]]] .= xleft[jmpidx[sortedidx[1:j]]]  
		mu  = b
	else
  		#x .= [LpProx(z[i], mu, p, proxtol)[1] for i=1:n] 
		if (ra - 1.0 < 1.0 - rb )
			x  = xleft
			mu = a
		else
			x  = xright
			mu = b
		end
	end
	#mu = dot(z - x, x)
  else	
  	mu = (a + b) / 2
  	x = [LpProx(z[i], mu, p, proxtol)[1] for i=1:n] 
  end
  # back to original scale
  x .= sign.(y) .* (r * x)
  #mu = exp(Base.log(mu) - (p - 2) * Base.log(r) )  # mu = mu * r / r^(p - 1)
  #return (x, numiters)
  log && IterativeSolvers.shrink!(history)
  x, mu, history
end

"""Projects y onto the nearest know case ball."""
function NearestExactProjection(y::Vector{T}, p::T, r = one(T)) where T <: Real
  if norm(y, p) <= r
    return (y, 0)
  end
  if p > 4.0 # infinity ball
    x = clamp.(y, -r, r)
  elseif p > 1.5 # Euclidean ball
    x = (r / norm(y, 2)) * y
  elseif p > 0.5 # L1 ball
    x = sign.(y) .* SimplexProjection(abs.(y), r)
  else # L0 ball
    x = zeros(T, length(y))
    perm = sortperm(y, by = abs, rev = true)
    (s, rp) = (zero(T), r^p)
    for j = 1:length(y)
      s = s + abs(y[perm[j]])^p
      if s > rp
        break
      else
        x[perm[j]] = y[perm[j]]
      end      
    end
  end
  return ((r / norm(x, p)) * x, 1)
end

"""Calculates the p norm of the optimal point for a given mu."""
function LpRadius(z::Vector{T}, mu::T, p::T, tol::T) where T <: Real
  n = length(z)
  r = zero(T)
  for i = 1:n
    (x, iters) = LpProx(z[i], mu, p, tol)
    r = r + abs(x)^p
  end
  return r^(1 / p)
end

"""Calculates the proximal map of (mu / p) |x|^p at y."""
function LpProx(y::T, mu::T, p::T, tol::T) where T <: Real
  if p >= one(T) 
    q = 1 / (1 - 1 / p) # conjugate exponent
    if p > q
        (x, iters) = NewtonProx(y, mu, p, tol)
        return (x, iters)
    else
        (z, iters) = NewtonProx(y / mu, 1 / mu, q, tol)
		x = y - mu * z   # Moreau decomposition
		if abs(x) < tol  # very close to zero, adjust
		  x = sign(y) * (abs(y) / mu)^(1 / (p - 1) )
		end
        return (x, iters)  
    end
  else
	if ( mu == zero(T) ) 
		return (y, 0)
	end
    z = abs(y)
    xm = ((1 - p) * mu)^(1 / (2 - p))  # minimizer of f'(x)
    fprime = xm - z + mu * abs(xm)^(p - 1) # f'(xm)
    if fprime < zero(T)	# min of f(x) on the right of xm
      (xn, iters) = NewtonProx(z, mu, p, tol)  # root of f'(x0) right of xm
      if (z - xn)^2 / 2 + (mu / p) * abs(xn)^p <= z^2 / 2	# f(xn) vs f(0)
        return (xn * sign(y), iters)
      else
        return (zero(T), iters)
      end
    else # f'(x) >= 0 for all x; f(x) is increasing -- min f(x) at x=0
      return (zero(T), 0)    
    end
  end
end

"""Calculates the proximal map of (mu / p) |x|^p at y for p >= 2
and p < 1 by Newton's method."""
function NewtonProx(y::T, mu::T, p::T, tol::T) where T <: Real
  p1 = p - one(T)
  if mu * (abs(y))^p1 < tol
    return (y, 0)
  end
  p2 = p1 - one(T) 
  x = sign(y) * min(abs(y), (abs(y) + mu * p2) / (1 + mu * p1))
  for iter = 1:100
    f = x - y + mu * abs(x)^p1 * sign(x)
    if abs(x) > one(T) || p < one(T)
      a = y / abs(x)^p2 + mu * p2 * x
   	  b = one(T) / abs(x)^p2 + mu * p1 
	else
   	  a = y + mu * p2 * abs(x)^p1 * sign(x)
   	  b = one(T) + mu * p1 * abs(x)^p2
	end
    x = a / b
    if abs(f) < tol # derivative test for convergence
      return (x, iter)
    end
  end
  return (x, 100)
end


"""Projects y onto the simplex with radius r."""
function SimplexProjection(y::Vector{T}, r::T) where T <: Real
  n = length(y)
  z = sort(y, rev = true)
  (s, lambda) = (zero(T), zero(T))
  for i = 1:n
    s = s + z[i]
    lambda = (s - r) / i
    if i < n && lambda < z[i] && lambda >= z[i + 1]
      break
    end
  end
  return max.(y .- lambda, zero(T))
end

"""
Projects y onto a weight simplex sum(w .* x) = r, x >= 0.
Sort-based.
""" function WeightedSimplexProjectionSort(y::Vector{T}, w::Vector{T}, r::T) where T <: Real n = length(y)
  z = y ./ w
  pidx = sortperm(z, rev = true) # descending
  (s, Wcum, lambda) = (zero(T), zero(T), zero(T))
  i = 1
  lambda = zero(T)
  while i <= n
    s    += w[pidx[i]] * y[pidx[i]]
	Wcum += w[pidx[i]]^2
    lambdanew = (s - r) / Wcum
#println("i = ", i, ", pidx[i] = ", pidx[i], ", s = ", s, ", Wcum = ", Wcum, ", lambda = ", lambda, ", lambdanew = ", lambdanew)
	(lambdanew > z[pidx[i]]) && break
	lambda = lambdanew
	i    += 1
  end
  return (max.(y .- lambda * w, zero(T)), lambda)
end

"""Projects the point y onto the weighted ell_1 ball with the given 
radius."""
function WeightedL1BallProjectionSort(y::Vector{T}, w::Vector{T},
  									  r::T, tol::T) where T <: Real
	x, lambda = WeightedSimplexProjectionSort(abs.(y), w, r)
	return sign.(y) .* x, lambda
end

"""
Projects y onto a weighted simplex sum(w .* x) = r, x >= 0.
Use Condat's (2017) accelerated Newton (Michelot) algorithm
"""
function WeightedSimplexProjectionCondat(y::Vector{T}, w::Vector{T},
  									  r::T) where T <: Real
	aux  = Vector{T}(undef, length(y))	# store both v_tilde and v
					# first `vtildeend` elements are v_tilde
					# next `auxlength - vtildeend` elements are v
	waux = Vector{T}(undef, length(w))	# store weights, follwing aux
	auxlength = 0   # actual number of stored values in aux
	vtildeend = 0	# pointer to the end of v_tilde in aux
	aux[1]  = y[1]
	waux[1] = w[1]
	auxlength += 1

	W2 = w[1]^2		# cumulative sum of squares of weights in v
	#rho = y[1] - r
	rho = (w[1]*y[1] - r) / W2
	#
	# begin 1st pass
	#
	for i = 2:length(y)
		if y[i] / w[i] > rho
			# add (y[i], w[i]) to the end of aux (and waux)
			aux[auxlength + 1]  = y[i]
			waux[auxlength + 1] = w[i]
			#rho += (y[i] - rho) / (auxlength - vtildeend + 1)
			rho += w[i] * (y[i] - w[i] * rho) / (W2 + w[i]^2)
			#if rho <= (y[i] - r) / w[i] 
			if rho <= (w[i] * y[i] - r) / w[i]^2
				# add v to v_tilde; set v = {y[i]}; reset W2 and rho
				vtildeend = auxlength 
				W2        = zero(T)
				#rho       = y[i] - r
				rho       = (w[i] * y[i] - r) / w[i]^2
			end
			auxlength += 1	# |v_tilde| + |v| increased by 1
							# if rho > (y[i] -r) / w[i], then
							#  y[i] has been added to v already
			W2 += w[i]^2
		end
	end
	if vtildeend > 0 	# v_tilde is nonempty 
		auxlength -= vtildeend	# |v|
		vtildeidx = vtildeend  # |v_tilde|
		while vtildeidx > 0	# for every elements of v_tilde
			yv = aux[vtildeidx]
			wv = waux[vtildeidx]
			if yv / wv > rho
				# move yv from v_tilde to v 
				auxlength += 1
				#rho += (yv - rho) / auxlength
				W2  += wv^2
				rho += wv * (yv - rho * wv) / W2
				aux[vtildeend]  = yv # prepend y to v
				waux[vtildeend] = wv # associated weight
				vtildeend -= 1 	   # expand v by 1 to the left
			end
			vtildeidx -= 1
		end
	end
	#
	# begin 2nd pass
	#
	while true
		auxlengthold = auxlength  # store previous value
		auxlength    = 0		  # number of elements in v
								  #  among aux seen so far
		for i = 1:auxlengthold
			yv = aux[vtildeend + i]	# y in v
			wv = waux[vtildeend + i] # associated weight
			if yv / wv > rho    
				# move y forward
				aux[vtildeend + auxlength + 1]  = yv
				waux[vtildeend + auxlength + 1] = wv
				auxlength += 1
			else
				# remove y from v; update rho
				# auxlengthold - i = number of elements in v unseen
				#rho += (rho - yv) / (auxlengthold - i + auxlength)
				#W2  -= wv^2	# catastrophic cancellation may occur
				#rho += wv * (wv * rho - yv) / W2
				wdiff = W2 - wv^2
				if abs(wdiff) < 1e-12   # catastrophic cancellation
					W2 = zero(T)
					s  = zero(T)
					for k = 1:auxlengthold
						(k == i) || (W2 += waux[vtildeend + k]^2, s += waux[vtildeend + k] * aux[vtildeend + k])
					end
					rho = (s - r) / W2
				else
					W2 = wdiff
					rho += wv * (wv * rho - yv) / W2
				end
			end
		end
		(auxlength < auxlengthold) || break
	end
    #@. aux = max(y - rho, zero(T))
    @. aux = max(y - rho * w, zero(T))
#println("aux = ", aux)
	return aux, rho
end

"""
Projects y onto a weighted L1 ball with radius r.
Use Condat's (2017) accelerated Newton algorithm.
"""
function WeightedL1BallProjectionCondat(y::Vector{T}, w::Vector{T},
  									  r::T) where T <: Real
	x, lambda = WeightedSimplexProjectionCondat(abs.(y), w, r)
	return sign.(y) .* x, lambda
end

"""
Projects y onto a nonconvx Lp ball with radius r.
Use the iteratively reweighted L1 ball projection (IRBP) 
by X. Yang, J. Wang, and H. Wang (2021)
"""
function IRBP1(y::Vector{T}, p::T, r::T;
			  maxiter = 1000,
			  tau::T=convert(T, 1.1),
			  M::T=convert(T, 100.0),
			  tol::T=convert(T, 1e-8),
			  log::Bool = false
			 ) where T <: Real
	n, pnorm = length(y), norm(y, p)
	if pnorm <= r
		return (y, zero(T), 0)
	end
	z = sign.(y) .* (y / r) # convert to nonnegative entries and radius 1
	# initialization
	x  = zeros(T, n)
	lambda = zero(T)
	nu = rand(n)
	e  = 0.9 * (nu / norm(nu, 1)).^(1 / p)
	alph0  = sum(abs.((z .- x) .* x - lambda * p * x.^p))
	bet0   = abs(sum(x.^p) - 1.0)

	history = ConvergenceHistory(partial = !log)
	history[:tol] = tol
	IterativeSolvers.reserve!(T, history, :alph, maxiter)
	IterativeSolvers.reserve!(T, history, :bet, maxiter)
	IterativeSolvers.reserve!(Float64, history, :itertime, maxiter)
	IterativeSolvers.nextiter!(history)

	for iteration = 1:maxiter
		tic = time()
		# stopping criterion
		alph  = sum(abs.((z .- x) .* x - lambda * p * x.^p))
		bet   = abs(sum(x.^p) - 1.0)
		push!(history, :alph,  alph)
		push!(history, :bet,   bet)
		if max(alph / n, bet / n) < tol * max(alph0 / n, bet0 / n, 1.0)
			toc = time()
			push!(history, :itertime, toc - tic)
			IterativeSolvers.setconv(history, true)
			break
		end

		#w = p * (x + e).^(p - 1)   # numerically unstable
		w = p * one(T) ./ ((abs.(x) + e).^(1 - p) .+ 1e-12)

		rho = one(T) - sum((x + e).^p) + sum(w .* x)
		## Condat's method: peril of catastrophic cancellation
		xnew, myrho = WeightedL1BallProjectionCondat(z, w, rho)
		# NaN occurs if `w` is practically unbounded.
		# In this case the corresponding x must be zero
		xnew[isnan.(xnew)] .= zero(T)
		act_ind = xnew .> zero(T)
		if any(act_ind)
			z_act_ind_outer = z[act_ind]
			x_act_ind_outer = xnew[act_ind]
			w_act_ind_outer = w[act_ind]
			# dual variable
			lambda = sum(z_act_ind_outer - x_act_ind_outer) / sum(w_act_ind_outer)
		end
		xdiff = xnew - x
		if norm(xdiff) * norm(sign.(xdiff) .* w)^tau <= M
			theta = min(bet, 1.0 / sqrt(iteration))^(1 / p)
			e *= theta
		end	
		x .= xnew

		toc = time()
		push!(history, :itertime, toc - tic)
		IterativeSolvers.nextiter!(history)
	end
	# dual variable
	mu = dot(z - x, x) / norm(x, p)^p  
	# back to original scale
	x .= sign.(y) .* (r * x)

	log && IterativeSolvers.shrink!(history)
	x, mu, history
end


"""
Projects y onto a nonconvx Lp ball with radius r.
Use the iteratively reweighted L1 ball projection (IRBP) 
by X. Yang, J. Wang, and H. Wang (2021)
"""
function IRBP2(y::Vector{T}, p::T, r::T;
			  maxiter = 1000,
			  tau::T=convert(T, 1.1),
			  M::T=convert(T, 100.0),
			  tol::T=convert(T, 1e-8),
			  log::Bool = false
			 ) where T <: Real
	n, pnorm = length(y), norm(y, p)
	if pnorm <= r
		return (y, zero(T), 0)
	end
	z = sign.(y) .* (y / r) # convert to nonnegative entries and radius 1
	# initialization
	x  = zeros(T, n)
	lambda = zero(T)
	nu = rand(n)
	e  = 0.9 * (nu / norm(nu, 1)).^(1 / p)
	alph0  = sum(abs.((z .- x) .* x - lambda * p * x.^p))
	bet0   = abs(sum(x.^p) - 1.0)

	history = ConvergenceHistory(partial = !log)
	history[:tol] = tol
	IterativeSolvers.reserve!(T, history, :alph, maxiter)
	IterativeSolvers.reserve!(T, history, :bet, maxiter)
	IterativeSolvers.reserve!(Float64, history, :itertime, maxiter)
	IterativeSolvers.nextiter!(history)

	for iteration = 1:maxiter
		tic = time()
		# stopping criterion
		alph  = sum(abs.((z .- x) .* x - lambda * p * x.^p))
		bet   = abs(sum(x.^p) - 1.0)
		push!(history, :alph,  alph)
		push!(history, :bet,   bet)
		if max(alph / n, bet / n) < tol * max(alph0 / n, bet0 / n, 1.0)
			toc = time()
			push!(history, :itertime, toc - tic)
			IterativeSolvers.setconv(history, true)
			break
		end

		#w = p * (x + e).^(p - 1)   # numerically unstable
		w = p * one(T) ./ ((abs.(x) + e).^(1 - p) .+ 1e-12)

		rho = one(T) - sum((x + e).^p) + sum(w .* x)
		## Duchi's method: slower but more accurate
		xnew, myrho = WeightedL1BallProjectionSort(z, w, rho, 1e-12)
		# NaN occurs if `w` is practically unbounded.
		# In this case the corresponding x must be zero
		xnew[isnan.(xnew)] .= zero(T)
		act_ind = xnew .> zero(T)
		if any(act_ind)
			z_act_ind_outer = z[act_ind]
			x_act_ind_outer = xnew[act_ind]
			w_act_ind_outer = w[act_ind]
			# dual variable
			lambda = sum(z_act_ind_outer - x_act_ind_outer) / sum(w_act_ind_outer)
		end
		xdiff = xnew - x
		if norm(xdiff) * norm(sign.(xdiff) .* w)^tau <= M
			theta = min(bet, 1.0 / sqrt(iteration))^(1 / p)
			e *= theta
		end	
		x .= xnew

		toc = time()
		push!(history, :itertime, toc - tic)
		IterativeSolvers.nextiter!(history)
	end
	# dual variable
	mu = dot(z - x, x) / norm(x, p)^p  
	# back to original scale
	x .= sign.(y) .* (r * x)

	log && IterativeSolvers.shrink!(history)
	x, mu, history
end


