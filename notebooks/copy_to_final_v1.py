import os
import shutil

def main():
    print("=== COPYING LOCKED MODEL A WORK TO final_v1 ===")
    
    # Destination base
    dest_base = "final_v1"
    os.makedirs(dest_base, exist_ok=True)
    
    # Define files and their destination subfolders
    files_to_copy = [
        # Models and results
        ("models/final/model_A_final_v1.json", "models"),
        ("models/final/model_A_final_v1.pkl", "models"),
        ("models/final/robust_xgb_optuna_results.json", "models"),
        ("models/final/robust_opt_plot_1_actual_vs_predicted.png", "models"),
        ("models/final/robust_opt_plot_4_residual_vs_time.png", "models"),
        
        # Configurations
        ("configs/model_A_features.json", "configs"),
        ("configs/model_B_features.json", "configs"),
        
        # Reports
        ("reports/model_A_feature_importance.csv", "reports"),
        ("reports/model_A_final_summary.md", "reports"),
        
        # Code & Scripts
        ("inference/predict_c4h8.py", "inference"),
        ("inference/predict_c4h6.py", "inference"),
        ("inference/predict_total_c4.py", "inference"),
        ("notebooks/verify_anchor_leakage.py", "notebooks"),
        ("notebooks/model_b_inversion_check.py", "notebooks"),
        
        # Documentation
        ("docs/01_problem_statement.md", "docs"),
        ("docs/02_dataset_understanding.md", "docs"),
        ("docs/03_feature_engineering.md", "docs"),
        ("docs/04_model_a_development.md", "docs"),
        ("docs/05_model_b_development.md", "docs"),
        ("docs/06_drift_analysis.md", "docs"),
        ("docs/07_final_architecture.md", "docs"),
        ("docs/08_deployment_guide.md", "docs"),
        ("docs/09_future_optimizer.md", "docs")
    ]
    
    for src_path, dest_subfolder in files_to_copy:
        dest_folder = os.path.join(dest_base, dest_subfolder)
        os.makedirs(dest_folder, exist_ok=True)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_folder)
            print(f"Copied {src_path} -> {dest_folder}/")
        else:
            print(f"Warning: Source file {src_path} not found!")

    # Write a quick info README inside final_v1
    readme_path = os.path.join(dest_base, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Model A (C4H8) Production Release v1\n\n")
        f.write("This folder contains all the frozen, validated components for the Model A soft-sensor.\n\n")
        f.write("## Folder Structure\n")
        f.write("*   `models/` — Frozen model binaries (.json, .pkl) and diagnostic validation plots.\n")
        f.write("*   `configs/` — Production feature configuration JSON.\n")
        f.write("*   `reports/` — Feature importance CSV and final summary report.\n")
        f.write("*   `inference/` — Live DCS prediction script with fallback logic.\n")
        f.write("*   `notebooks/` — Formal leak-free target validation script.\n")
        
    print(f"Created release README.md in {dest_base}/")
    print("=== COPY TO final_v1 COMPLETE ===")

if __name__ == "__main__":
    main()
