
import os
import subprocess
from google.colab import userdata

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

dagshub_token = userdata.get('dagshub_token')

run("dvc remote modify storage --local auth basic")
run("dvc remote modify storage --local user hmelbanna100")
run(f"dvc remote modify storage --local password {dagshub_token}")

# Pull latest DVC data
run("dvc pull")

# Pull latest Git changes
run("git pull")

print("Setup completed successfully.")
