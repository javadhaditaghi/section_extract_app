import subprocess
import sys
import os

python_executable = sys.executable

print("🚀 Running annotator.py...")
subprocess.run([python_executable, "annotator.py"], check=True)

print("🔄 Running post_processing.py...")
subprocess.run([python_executable, "post_processing.py"], check=True)

print("✅ Annotation pipeline completed.")
