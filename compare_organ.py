#!/usr/bin/env python3
import argparse
import os
import subprocess
import uuid
from datetime import datetime

def compare_organ(ct_path, mask1_path, mask2_path, organ,
                  base_output="./comparison_results", base_csv="./results",
                  port="8000", log_file="./comparison_summary.log", run_id = None):
    """
    Compare two segmentation masks using vLLM pipeline and write a detailed log.
    """
    if run_id is None:
        run_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join(base_output, f"{run_id}", f"{organ}")
    csv_dir = os.path.join(base_csv, f"{run_id}")
    csv_path = os.path.join(csv_dir, f"{organ}.csv")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Step 1. Projection
    result = subprocess.run([
        "python3", "ProjectDatasetFlex_single.py",
        "--ct_good", ct_path,
        "--mask_good", mask1_path,
        "--ct_bad", ct_path,
        "--mask_bad", mask2_path,
        "--organ", organ,
        "--output_dir", output_dir
    ], capture_output=True, text=True)
    print("Projection STDOUT:\n", result.stdout)
    print("Projection STDERR:\n", result.stderr)
    result.check_returncode()
    if True:
        # Step 2. Run API
        subprocess.run([
            "python3", "RunAPI_single.py",
            "--path", output_dir,
            "--organ", organ,
            "--csv_path", csv_path,
            "--port", str(port)
        ], check=True)

        # Step 3. Parse result
        better = None
        with open(csv_path, "r") as f:
            header = f.readline().strip().split(",")
            row = f.readline().strip().split(",")
            if len(row) >= 2:
                answer = row[1].strip()
                if answer == "1":
                    better = "mask1"
                elif answer == "2":
                    better = "mask2"
                elif answer == "0.5":
                    better = "mask1"
                    print("Undecided case (answer=0.5):", ",".join(row))

        # Determine which path is better
        best_path = mask1_path if better == "mask1" else mask2_path

        # Step 4. Write to log file
        with open(log_file, "a") as log:
            log.write(
                f"[{datetime.now().isoformat()}] Run ID: {run_id}\n"
                f"  Organ: {organ}\n"
                f"  CT: {ct_path}\n"
                f"  Mask1: {mask1_path}\n"
                f"  Mask2: {mask2_path}\n"
                f"  Better: {best_path}\n\n"
            )
        print(mask1_path, mask2_path)
        print(best_path)
    else:
        best_path = mask1_path  # Default to mask1 if not running API
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two segmentation masks using vLLM and log the result.")
    parser.add_argument("--ct", required=True, help="Path to CT volume (.nii.gz)")
    parser.add_argument("--mask1", required=True, help="Path to segmentation directory for model 1")
    parser.add_argument("--mask2", required=True, help="Path to segmentation directory for model 2")
    parser.add_argument("--organ", required=True, help="Organ name (e.g. aorta, pancreas)")
    parser.add_argument("--base_output", default="./comparison_results", help="Base folder for projection results")
    parser.add_argument("--base_csv", default="./results", help="Base folder for CSV results")
    parser.add_argument("--port", default="8000", help="API server port (vLLM)")
    parser.add_argument("--log_file", default="./comparison_summary.log", help="File to append final comparison results")
    args = parser.parse_args()

    best = compare_organ(
        args.ct, args.mask1, args.mask2, args.organ,
        base_output=args.base_output, base_csv=args.base_csv,
        port=args.port, log_file=args.log_file
    )
    print(best)
