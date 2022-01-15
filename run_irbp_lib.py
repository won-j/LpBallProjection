#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:48:48 2021

@author: Xiangyu Yang
@modified by: Joong-Ho (Johann) Won, 2022-01
    - returns no. iterations
"""
import numpy as np
from numpy import linalg as LA

import irbp_lib



def get_sol(point_to_be_projected, p, radius) -> float:
    """Test irbp.

    Args:
        point_to_be_projected: Point to be projected.
        p: p parameter for lp-ball.
        radius: The radius of lp-ball.

    Returns:
        x_irbp : The projection point.
    """
    
    data_dim = point_to_be_projected.shape[0]
    #%% Initialization
    x_ini = np.zeros(data_dim, dtype= np.float64)
        
    rand_num = np.random.uniform(0, 1, data_dim)
    rand_num_norm = LA.norm(rand_num, ord=1)
    epsilon_ini = 0.9 * (rand_num * radius / rand_num_norm) ** (1. / p) # ensure that the point is feasible.
        
    x_irbp, dual, iter  = irbp_lib.get_lp_ball_projection(x_ini, point_to_be_projected, p, radius, epsilon_ini)
        
    return x_irbp, dual, iter
