#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 09:58:49 2021

@author: Xiangyu Yang
@modified by Joong-Ho (Johann) Won, 2022-01
    - error handling
    - returns no. iterations
"""

import numpy as np
from numpy import linalg as LA

import alternating_proj_lib

def get_lp_ball_projection(starting_point, point_to_be_projected, p, radius, epsilon) -> float:
    
    """Gets the lp ball projection of given point.

    Args:
        point_to_be_projected: Point to be projected.
        starting_point: Starting point of IRBP.
        p: p parameter for lp-ball.
        radius: The radius of lp-ball.
        epsilon: Initial value of the smoothing parameter epsilon.

    Returns:
        x_final : The projection point.
        dual : The dual variable.
    """

    # Step 1 and 2 in IRBP. Input parameters : note that our tolerance parameter are set as 1e-8
    n = point_to_be_projected.shape[0]

    tau, tol = 1.1, 1e-8
    max_iteration_num = int(1e3)  # The maximal number of iterations
    condition_right = 100.0 # the right term of the condition for updating epsilon

    signum = np.sign(point_to_be_projected) 
    bar_y = signum * point_to_be_projected  # bar_y lives in the positive orthant of R^n
    
    cnt = 0  

    x_final = np.zeros(n)

    # store the info of residuals
    res_alpha = []  # the optimal residual
    res_beta = []  # the feasibility residual
    
    
    if LA.norm(point_to_be_projected, p) ** p <= radius:  
        return None
    
    # The loop of IRBP
    while True:

        cnt += 1

        # Step 3 in IRBP. Compute the weights
        weights = p * 1. / ((np.abs(starting_point) + epsilon) ** (1 - p) + 1e-12)

        # Step 4 in IRBP. Solve the subproblem for x^{k+1}
        gamma_k = radius - LA.norm(abs(starting_point) + epsilon, p) ** p + np.inner(weights, abs(starting_point))

        assert gamma_k > 0, "The current Gamma is non-positive"

        # Subproblem solver : The projection onto weighted l1-ball
        x_new = alternating_proj_lib.get_weightedl1_ball_projection(bar_y, weights, gamma_k)

        #print('iter = ', cnt, ', x = ', starting_point)
        #print('        e = ', epsilon)
        #print('        w = ', weights)
        #print('\txnew = ', x_new)
        
        local_reduction = x_new - starting_point
        
        # Note that if x_new is not feasible to lp-ball, then we do a truncation using the last iterate

        if LA.norm(x_new + epsilon, p) ** p > radius:
            large_ = np.nonzero(local_reduction >= 0)
            large_ = large_[0]
            x_new[large_] = starting_point[large_]  # Truncation operation

        # Step 5 in IRBP. Set the new relaxation vector epsilon according to the proposed condition
        condition_left = LA.norm(local_reduction, 2) * LA.norm(np.sign(local_reduction) * weights, 2) ** tau
        error_appro = abs(LA.norm(x_new, p) ** p - radius)  

        if condition_left <= condition_right:
            theta = np.minimum(error_appro ** (1. / p), (1. / np.sqrt(cnt)) ** (1. / p))
            epsilon = theta * epsilon

        act_ind = x_new > 0 # Nonempty active index set

        # Compute dual iterate and two residuals
        if any(act_ind):
            y_act_ind_outer = bar_y[act_ind]
            x_act_ind_outer = x_new[act_ind]  
            weights_ind_outer = weights[act_ind]
    
           
            lambda_new = np.divide(np.sum(y_act_ind_outer - x_act_ind_outer), np.sum(weights_ind_outer))
    
            residual_alpha = (1. / n) * LA.norm((bar_y - x_new) * x_new - p * lambda_new * x_new ** p, 1)
            residual_beta = (1. / n) * error_appro
    
            res_alpha.append(residual_alpha)
            res_beta.append(residual_beta)
    
            # Step 6 in IRBP. Set k <--- (k+1)
            starting_point = x_new.copy()  
        else:  # no coordinate is active. return zero
            x_final = signum * x_new # symmetric property of lp ball
            dual = np.nan
            break

        # Check the stopping criteria
        if max(residual_alpha, residual_beta) < tol * max(max(res_alpha[0], res_beta[0]),
                                                              1) or cnt >= max_iteration_num:
            x_final = signum * x_new # symmetric property of lp ball
            dual = lambda_new
            break
        
    return x_final, dual, cnt
