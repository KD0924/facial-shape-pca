# -*- coding: utf-8 -*-
"""
Test functions relating to the similarity transform aligning two
sets of points.

@author: Tim Cootes
"""

import numpy as np
import tc_shape.sim_transform as sim_transform
 
def test_calc_sim_transform():
    """Test the calc sim_transform() function"""
    
    # Set up three points
    pts1=np.array([[0,0],[1,0],[0,4]])
    
    # Set up transformation parameters
    a=np.array([2,1, 1.2,3.4])

    # Transform the points
    pts2=sim_transform.apply(pts1,a)
    print("pts2\n",pts2)
    
    # Estimate transformation from pts1 to pts2
    a_est=sim_transform.calc(pts1,pts2)
    print("True transform     : ",a)
    print("Estimated transform: ",a_est)

    print("Error on estimate = {:.2f}".format(np.linalg.norm(a_est-a)))

    # Apply inverse to pts2
    a_inv=sim_transform.invert(a)
    pts3=sim_transform.apply(pts2,a_inv)
    print("Error on points after inverting = {:.2f}".format(np.linalg.norm(pts3-pts1)))
    
def test_invert_sim_transform():
    """Test the invert sim_transform function"""
    
    # Set up transformation parameters
    a=np.array([2,1, 1.2,3.4])
    b=sim_transform.invert(a)
    c=sim_transform.invert(b)
    
    print("Error on inv(inv(a))= {:.2f}".format(np.linalg.norm(c-a)))

# Run the tests
def run_tests():
  test_calc_sim_transform()
  test_invert_sim_transform()
