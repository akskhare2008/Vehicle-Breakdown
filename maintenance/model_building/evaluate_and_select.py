import mlflow
import pandas as pd
import os

# Set tracking URI to local directory where MLflow runs were saved
mlflow.set_tracking_uri('mlruns')

experiment_name = "vehicle_breakdown"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    print(f"Experiment '{experiment_name}' not found.")
    exit(1)

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

if runs.empty:
    print("No runs found in experiment.")
    exit(1)

# Identify the best model based on test_f1
best_run = runs.loc[runs['metrics.test_f1'].idxmax()]
best_model_name = best_run['params.model_name']

print(f"Best Model: {best_model_name} with F1: {best_run['metrics.test_f1']}")

# Save the best model name for subsequent steps if needed
with open('best_model.txt', 'w') as f:
    f.write(best_model_name)
