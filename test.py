import numpy as np
import matplotlib.pyplot as plt

def plot_arrow_to_origin(point_x, point_y, xlims, ylims, length=20):
    """
    Plot an arrow of specified length pointing toward (0,0) from a point outside plot bounds.
    
    Args:
        point_x, point_y: coordinates of the external point
        xlims, ylims: plot boundaries as [min, max] lists
        length: arrow length in plot units (default 20)
    """
    # Calculate angle from point to origin (0,0)
    angle_rad = np.arctan2(-point_y, -point_x)
    
    # Find intersection with plot boundary
    # Check which boundary the line from point to origin crosses first
    t_vals = []
    
    # Check x boundaries
    if point_x != 0:
        for x_bound in xlims:
            t = (x_bound - point_x) / (-point_x)
            if t > 0:
                y_intersect = point_y + t * (-point_y)
                if ylims[0] <= y_intersect <= ylims[1]:
                    t_vals.append(t)
    
    # Check y boundaries  
    if point_y != 0:
        for y_bound in ylims:
            t = (y_bound - point_y) / (-point_y)
            if t > 0:
                x_intersect = point_x + t * (-point_x)
                if xlims[0] <= x_intersect <= xlims[1]:
                    t_vals.append(t)
    
    # Use the smallest positive t (closest boundary intersection)
    t_min = min(t_vals) if t_vals else 0
    boundary_x = point_x + t_min * (-point_x)
    boundary_y = point_y + t_min * (-point_y)
    
    # Calculate arrow start point (length units inside the boundary)
    start_x = boundary_x - length * np.cos(angle_rad)
    start_y = boundary_y - length * np.sin(angle_rad)
    
    # Plot arrow
    plt.annotate('', xy=(boundary_x, boundary_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    return start_x, start_y, boundary_x, boundary_y