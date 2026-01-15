import os
import time

def run_step(script_name, step_name):
    print(f"\n{'='*60}")
    print(f">>> STEP {step_name}: Running {script_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    # Execute the python script
    exit_code = os.system(f"python src/{script_name}")
    end_time = time.time()
    
    if exit_code == 0:
        print(f"✅ SUCCESS: {script_name} finished in {end_time - start_time:.2f} seconds.")
    else:
        print(f"❌ ERROR: {script_name} failed. Pipeline stopped.")
        exit(1)

if __name__ == "__main__":
    print("\n🚀 STARTING FRAUD DETECTION PIPELINE")
    
    # 1. Person 1: Data
    run_step("data_processing.py", "1 (Data Cleaning)")
    
    # 2. Person 2: Graph
    run_step("graph_analysis.py", "2 (Graph Intelligence)")
    
    # 3. Person 3: ML
    run_step("ml_model.py", "3 (Model Training)")
    
    # 4. Person 4: Alerts
    run_step("explainability.py", "4 (Alert Generation)")
    
    print("\n🎉 PIPELINE COMPLETE! All results saved in /data folder.")