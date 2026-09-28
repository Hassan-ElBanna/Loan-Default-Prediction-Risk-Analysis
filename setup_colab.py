import os
import subprocess

def run(command):
    print(f">>> {command}")
    subprocess.run(command, shell=True, check=True)

# Install DVC
run("pip install dvc")

# Configure Git
run('git config --global user.name "Hassan"')
run('git config --global user.email "200016662@must.edu.eg"')

# Configure DVC remote
run("dvc remote remove storage || true")
run("dvc remote add -d storage https://dagshub.com/hmelbanna100/Loan_Default_Prediction_Risk_Analysis.dvc")

# Get DagsHub token from environment variable
dagshub_token = os.environ.get("DAGSHUB_TOKEN")

if not dagshub_token:
    raise ValueError(
        "DAGSHUB_TOKEN is not set. "
        "Set it before running this script."
    )

run("dvc remote modify storage --local auth basic")
run("dvc remote modify storage --local user hmelbanna100")
run(f"dvc remote modify storage --local password {dagshub_token}")

# Pull latest DVC data
run("dvc pull")

# Pull latest Git changes
run("git pull")

print("Setup completed successfully.")
