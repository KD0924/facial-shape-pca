# -*- coding: utf-8 -*-
"""
Utility functions for plotting points/shapes.

Represent shapes as set of points {x0,y0,x1,y1...}

Define lists of integers indicating how to connect the points 
to draw a curve.  So curve=[0,1,2,6,7].
To close the curve, last point index should be the same as the first.

curves is a list of such lists.

@author: Tim Cootes
"""

# Plot the points (assumed stored as single row {x0,y0,x1,y1...})
def plot_points(plt, points, style):
    pts=points.reshape(-1,2)    
    # y runs from top in images, so invert when plotting.
    plt.plot(pts[:,0],-pts[:,1],style)
    
def plot_curves(plt, points, curves, style):
    """For each curve in curves, plot lines to join points.
    curves is list of lists of integers."""
    pts=points.reshape(-1,2)
    for c in curves:
        plt.plot(pts[c,0],-pts[c,1],style)
    
