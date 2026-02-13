import os
import subprocess
import sys

def run_step(step_name, command):
    print("\n" + "="*70)
    print(f"🚀 RUNNING STEP: {step_name}")
    print("="*70)
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"❌ Step failed: {step_name}")
        sys.exit(1)
    else:
        print(f"✅ Step completed: {step_name}")

def main():
    print("""
    ================================================
      EXCEL TABLE DETECTION - END TO END PIPELINE
    ================================================
    """)

    # 1. Feature Extraction
    run_step(
        "Feature Extraction (Structural + Semantic)",
        "python3 features/feature_extractor.py"
    )

    # 2. Train & Compare Models
    run_step(
        "Model Training & Comparison (MLflow Logging)",
        "python3 model/train_model.py"
    )

    # 3. Start MLflow UI (Optional)
    print("\n📊 To view experiments, run:")
    print("   mlflow ui")

    print("\n🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
