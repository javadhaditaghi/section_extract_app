import subprocess

print("🚀 Running annotator.py...")
subprocess.run(["python", "annotator.py"], check=True)

print("🔄 Running post_processing.py...")
subprocess.run(["python", "post_processing.py"], check=True)

print("✅ Annotation pipeline completed.")
