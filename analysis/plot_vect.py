#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

def read_vect(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    
    # Format:
    # 1 (number of components)
    # N (number of points)
    # x y z ...
    
    n_pts = int(lines[1])
    pts = []
    for line in lines[2:2+n_pts]:
        pts.append([float(x) for x in line.split()])
    return np.array(pts)

def main():
    if len(sys.argv) < 3:
        print("Usage: plot_vect.py <input.vect> <output.png>")
        sys.exit(1)
        
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    
    pts = read_vect(in_path)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Close the loop
    pts_closed = np.vstack([pts, pts[0]])
    
    ax.plot(pts_closed[:,0], pts_closed[:,1], pts_closed[:,2], '-', linewidth=3, color='#1f77b4')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='red', s=5)
    
    # Keep aspect ratio equal
    max_range = np.array([pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), pts[:,2].max()-pts[:,2].min()]).max() / 2.0
    mid_x = (pts[:,0].max()+pts[:,0].min()) * 0.5
    mid_y = (pts[:,1].max()+pts[:,1].min()) * 0.5
    mid_z = (pts[:,2].max()+pts[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D render to {out_path}")

if __name__ == '__main__':
    main()
