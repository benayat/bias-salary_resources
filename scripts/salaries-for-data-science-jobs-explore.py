import subprocess
import sys

# Run the first script
result1 = subprocess.run([sys.executable, 'load_salaries_and_process.py'], cwd='scripts')
if result1.returncode != 0:
    print("Error in load_salaries_and_process.py")
    sys.exit(1)

# Run the second script
result2 = subprocess.run([sys.executable, 'classify_salaries_ai_ml.py'], cwd='scripts')
if result2.returncode != 0:
    print("Error in classify_salaries_ai_ml.py")
    sys.exit(1)

print("All scripts completed successfully.")
