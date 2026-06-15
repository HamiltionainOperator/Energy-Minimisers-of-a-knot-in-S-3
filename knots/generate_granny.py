import numpy as np
import os
import argparse

def generate_granny_knot(N=2000):
    """
    Parametric Fourier equation for the Granny Knot (3_1 # 3_1)
    Derived by Aaron Trautwein (1995).
    """
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    x = -0.22 * np.cos(t) - 1.28 * np.sin(t) - 0.44 * np.cos(3*t) - 0.78 * np.sin(3*t)
    y = -0.10 * np.cos(2*t) - 0.27 * np.sin(2*t) + 0.38 * np.cos(4*t) + 0.46 * np.sin(4*t)
    z =  0.70 * np.cos(3*t) - 0.40 * np.sin(3*t)
    
    # Scale it up to match our spatial grid
    scale = 20.0
    return np.column_stack([x, y, z]) * scale

def write_vect(filename, pts):
    with open(filename, 'w') as f:
        f.write("1\n")
        f.write(f"{len(pts)}\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

def write_obj(filename, pts):
    with open(filename, 'w') as f:
        for p in pts:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        
        f.write("l")
        for i in range(1, len(pts)+1):
            f.write(f" {i}")
        f.write(" 1\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a composite Granny knot")
    parser.add_argument('--n', type=int, default=2000, help="Number of vertices")
    parser.add_argument('--out', type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    pts = generate_granny_knot(args.n)
    
    # Write geometry files
    write_vect(os.path.join(args.out, 'Granny.vect'), pts)
    write_obj(os.path.join(args.out, 'Granny.obj'), pts)
    
    print(f"Generated Composite Granny Knot (N={args.n}) in {args.out}")
