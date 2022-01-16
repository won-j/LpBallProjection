# Convex and nonconvex Lp-ball Projection

Code for reproducing results in "A unified analysis of convex and non-convex lp-ball projection problems" by Joong-Ho Won, Kenneth Lange, and Jason Xu (under revision)

Implements of numerical algorithms for solving the Euclidean projection of a vector onto a convex or nonconvex lp-ball: 

```
minimize    0.5|| x - y ||_2^2
subject to  || x ||_p <= r
```

where 0 < p < `Inf`.

## Implemented algorithms

1. Dual Newton (code 6): for p > 1. Algorithm proposed in the paper.
2. Dual bisection (code 7): for **all** p > 0. Algorithm proposed in the paper.
3. Naive method (code 1): choosing the $\ell_{\infty}$ ball for p > 4, the $\ell_2$ ball for 2 <= p <=4, the $\ell_1$ ball for 1/2 < p <= 3/2, and the $\ell_0$ ball for 0 < p <= 1/2.
4. Projected Newton (code 8): for p > 1. Taken from the [proxTV](https://github.com/albarji/proxTV) toolboxa.
5. Iteratively Re-weighted $\ell_1$-Ball Projection 1 (IRBP1, code 10): for p < 1. Method of [Towards an efficient approach for the nonconvex ℓp-ball projection: algorithm and analysis](https://arxiv.org/abs/2101.01350), by Xiangyu Yang and Jiashan Wang and Hao Wang. Uses the weighted version of the fast $\ell_1$-ball projection method by L. Condat, "Fast projection onto the simplex and the l1 ball," Mathematical Programming Series A, vol. 158, no. 1, pp. 575-585, July 2016.
6. IRBP 2 (code 11): for p < 1. Uses the weighted version of the sort-based $\ell_1$-ball projection method by Duchi, J., Shalev-Shwartz, S., Singer, Y., Chandra, T.: [Efficient projections onto the l1-ball for learning in high dimensions](https://web.stanford.edu/~jduchi/projects/DuchiShSiCh08.html).
7. IRBP 3 (code 12): for p < 1. [Reference implementation](https://github.com/Optimizater/Lp-ball-Projection) by Xiangyu Yang (in Python).

## Usage

0. Preparation 
	* Install [Julia](https://julialang.org) v1.5 or above.
	* Change directory to where this repo is cloned.
	* [Install packages](https://docs.julialang.org/en/v1/stdlib/Pkg/): 
		- Run Julia by
		```
		julia --project=.. 
		```
		to go into the REPL mode.
		- Install packages [DataFrames](https://github.com/JuliaData/DataFrames.jl), [CSV](https://github.com/JuliaData/CSV.jl), [IterativeSolvers](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl), [PyCall](https://github.com/JuliaPy/PyCall.jl), [LatexPrint](https://github.com/scheinerman/LatexPrint.jl) by typing
		```
		(v1.5) pkg> add DataFrames
		```
		etc in the package mode (press `]` in the REPL mode)..
	* Clone this repo.

1. Reproducing Table 1
	- Change directory to where this repo is cloned.
	- To be able to use the Projected Newton method, type
	```
	$ cd src
	$ make
	$ cd ..
	```
	(macOS is required; `$` is the shell prompt).
	- If you want to disable Projected Newton, comment out Lines 33, 34 of `testprojectionmaps_cvx.jl` and change Line 14 as
	```
	nummethods = 3
	```
	- Run
	```
	julia --project=.. testprojectionmaps_cvx.jl
	```

2. Reproducing Table 2
	- Change directory to where this repo is cloned.
	- To be able to use IRBP3, install Python3 ([Anaconda distribution](https://www.anaconda.com) is recommended).
	- If you want to disable IRBP, comment out Lines 8, 22-24, 42-45 of `testprojectionmaps_noncvx.jl` and change Line 16 as
	```
	nummethods = 4
	```
	- Run
	```
	julia --project=.. testprojectionmaps_noncvx.jl
	```

