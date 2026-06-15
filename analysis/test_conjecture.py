#!/usr/bin/env python3
import subprocess
import sys
import os

# Batch of Composite Knots to test (name, generator_type, expected_determinant)
# The determinant of a connected sum is the product of the components' determinants.
# Trefoil has det=3.
KNOTS = [
    ("Granny", "granny", 9),         
]

ITERS = 10000
N = 1000
STEP = 0.0001

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def stream_cmd(cmd):
    # Streams output directly to the terminal in real-time
    print(f"\n>> {cmd}")
    subprocess.run(cmd, shell=True)

def verify_knot(path, expected_det):
    # Robust determinant: majority vote over random rotations, so a degenerate
    # default projection can't mislabel a real knot as the unknot (see
    # analysis/knot_check.py). Returns (passed, det) or (False, "<value> (split)").
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from knot_check import read_vect_pts, robust_determinant
        pts = read_vect_pts(path)
        det, dist, unanimous = robust_determinant(pts, ntries=11)
        label = det if unanimous else f"{det} (split {dist})"
        return (det == expected_det), label
    except Exception as e:
        return False, str(e)

def main():
    print(f"Testing Conjecture 4.4 on {len(KNOTS)} COMPOSITE knot types (ITERS={ITERS}, N={N})...")
    print("="*80)
    print(f"{'Knot':<10} | {'Initial |g|':<15} | {'Final |g|':<15} | {'Minimizer Found? (Isotopy)'}")
    print("-" * 80)
    
    results_md = [
        "# Composite Knot Conjecture Test Results\n\n",
        f"Testing the existence of an $E_{{S^3}}^{{2,1}}$-minimizer across composite knot types ($N={N}$, Iters={ITERS}, $\\alpha_0={STEP}$).\n\n",
        "| Knot | Initial Gradient | Final Gradient | Minimizer Found? (Isotopy) |\n",
        "|---|---|---|---|\n"
    ]
    
    for knot_name, gen_type, expected_det in KNOTS:
        
        out_dir = f"output/{knot_name}"
        init_vect = f"{out_dir}/{knot_name}.vect"
        log_file = f"{out_dir}/energy_log.csv"
        s3_vect = f"{out_dir}/{knot_name}_s3.vect"
        
        # 1. Generate the composite knot directly
        run_cmd(f"python3 knots/generate_composites.py --n {N} --type {gen_type} --out {init_vect}")
        
        # 2. Stream the actual long-running minimization step so we see the iteration logs!
        stream_cmd(f"build/energy_s3 {init_vect} {log_file} {s3_vect} {ITERS} {STEP} --ccd")
        
        # 3. Generate plots
        run_cmd(f"python3 analysis/plot_energy.py {log_file} --out {out_dir}/energy_log.png --title '{knot_name} Knot on S3'")
        run_cmd(f"python3 analysis/plot_vect.py {s3_vect} {out_dir}/{knot_name}_render.png")
        
        # 2. Extract energies from the log file
        init_e = "N/A"
        final_e = "N/A"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.read().splitlines()
                if len(lines) > 1:
                    first_data = lines[1].split(',')
                    last_data = lines[-1].split(',')
                    init_e = f"{float(first_data[2]):.4f}"
                    final_e = f"{float(last_data[2]):.4f}"
                    
        # 3. Check invariants via pyknotid directly
        passed, det = verify_knot(s3_vect, expected_det)
        
        if passed:
            isotopy_pass = f"PASS ✅ (det={det})"
        else:
            isotopy_pass = f"FAIL ❌ (det={det}, exp={expected_det})"
            
        print(f"{knot_name:<10} | {init_e:<15} | {final_e:<15} | {isotopy_pass}", flush=True)
        results_md.append(f"| {knot_name} | {init_e} | {final_e} | {isotopy_pass} |\n")
        results_md.append("\n<details><summary><b>View Plots & Renders</b></summary>\n\n")
        results_md.append(f"![{knot_name} Energy Log](/Users/yash/knot-s3/{out_dir}/energy_log.png)\n")
        results_md.append(f"![{knot_name} 3D Render](/Users/yash/knot-s3/{out_dir}/{knot_name}_render.png)\n")
        results_md.append("\n</details>\n\n")
        
    print("="*80)
    
    with open("output/conjecture_results.md", "w") as f:
        f.writelines(results_md)
        
if __name__ == '__main__':
    main()
