"""
OncoPulse — Master Runner
=========================
Runs the full pipeline end-to-end.
Usage: python run_pipeline.py [simulate|geo]
"""

import sys
import subprocess
import os

def run(script: str, args: list = []):
    cmd = [sys.executable, script] + args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, check=True, capture_output=False)
    return result.returncode == 0

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "simulate"

    scripts = [
        ("stage1_data_ingestion.py", [mode]),
        ("stage2_preprocessing.py",  []),
        ("stage3_biomarker_discovery.py", []),
        ("stage4_ml_risk_model.py",  []),
    ]

    for script, args in scripts:
        if os.path.exists(script):
            success = run(script, args)
            if not success:
                print(f"[!] {script} failed. Fix errors before continuing.")
                sys.exit(1)
        else:
            print(f"[!] {script} not found — skipping")

    print("\n" + "="*60)
    print("✓ All stages complete!")
    print("Launch dashboard with: streamlit run stage5_dashboard.py")
    print("="*60)
