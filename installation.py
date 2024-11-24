import subprocess

with open('requirements.txt') as f:
    for line in f:
        package = line.strip()
        try:
            subprocess.check_call(['pip', 'install', package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}, continuing...")
