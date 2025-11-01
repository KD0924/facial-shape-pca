# -*- coding: utf-8 -*-
"""
Functions to find and use the similarity transform aligning two
sets of points.

@author: Tim Cootes
"""

import numpy as np
import math


def calc(pts1,pts2):
    """Calculate similarity transform mapping pts1 to pts2
    pts1: n x 2 array, so one point is (pts1[i,0],pts1[i,1])
    Returns four element array (a,b,tx,ty).
    Transformation is  new_x=a*x-b*y+tx, new_y=b*x+a*y+ty
    """
    c1=pts1.mean(0)
    c2=pts2.mean(0)
    d1=pts1-c1.T
    d2=pts2-c2.T
    
    S1=np.sum(np.square(d1))
    Sa=np.sum(d1*d2)
    Sb=np.dot(d1[:,0],d2[:,1])-np.dot(d1[:,1],d2[:,0])
    
    a=Sa/S1
    b=Sb/S1
    tx=c2[0]-(a*c1[0]-b*c1[1])
    ty=c2[1]-(b*c1[0]+a*c1[1])
    
    return np.array([a,b,tx,ty])
    
def apply(pts,a):
    """Transform points pts by similarity transform.
    pts1: n x 2 array, so one point is (pts1[i,0],pts1[i,1])
    Transformation is 
      new_x=a[0]*x-a[1]*y+a[2]
      new_y=a[1]*x+a[0]*y+a[3]
    Return result in same form as input (n x 2 matrix)
    """
     
    pts2=pts@np.array([[a[0],a[1]],[-a[1],a[0]]])
    pts2[:,0]+=a[2] # Translate x
    pts2[:,1]+=a[3] # Translate y
    return pts2


def invert(a):
    """Computes parameters of inverse of transform.
    Transform encoded as (a,b,tx,ty), Ax+t
    Inverse computed as inv(A)[x-t]
    """
    det=a[0]*a[0]+a[1]*a[1]
    
    # Inverse of scale/rotation part: 
    b1=a[0]/det
    b2=-a[1]/det
    tx= b1*a[2]-b2*a[3]
    ty= b2*a[2]+b1*a[3]
    return np.array([b1,b2,-tx,-ty])

def normalise(pts):
    """Return normalised points: CoG at origin, unit scale."""
    cog=pts.mean(0)
    pts1=pts-cog
    
    # Compute scale
    s2=np.sum(np.square(pts1))
    
    f=1.0/math.sqrt(max([1e-6,s2]))
    
    return f*pts1

def align_to_ref_pts(S,ref_pts):
    """Align each shape (row of S) to ref_pts
    Treats each row of S as shape (x0,y0,x1,y1,...)
    Aligns each to ref_pts and returns new matrix with 
    aligned shapes in the rows.
    """
    S2=S
    
    n_shapes=S.shape[0]
    
    # Align each shape into the frame of the ref_shape
    for i in range(n_shapes):
        # Get shape i
        pts=S[i,:].reshape(-1,2)
        
        # Find transform from ref_pts to pts
        a=calc(ref_pts,pts)
        
        # Invert this
        a_inv=invert(a)
        
        ## Apply to pts to map into reference frame
        pts2 = apply(pts,a_inv)
        
        # Place these points in the row of S2
        S2[i,:]=pts2.flatten()
        
    return S2
    
 
def align_set(S):
    """Align set of shapes encoded in rows of S as (x0,y0,x1,y1....)
    Return matrix whose rows are aligned shapes.
    """
    
    # Use first shape as reference initially
    ref_shape=S[0]
    
    # Convert to a nx2 matrix, each row one point
    ref_pts=ref_shape.reshape(-1,2)
    
    ref_pts=normalise(ref_pts)
    
    # Record initial example to define orientation
    ref_pts0=ref_pts
    
    n_its=3
    for i in range(n_its):
        S=align_to_ref_pts(S, ref_pts)
        mean_shape=S.mean(0)
        mean_pts=mean_shape.reshape(-1,2)
        
        # Align to original reference points to get orientation.
        a=calc(ref_pts0,mean_pts)
        mean_pts=apply(mean_pts,invert(a))
        
        # Normalise to ensure CoG at origin, unit scale
        ref_pts=normalise(mean_pts)
        
    return S