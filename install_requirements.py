# NOT TESTED! Might not work

import subprocess
import sys

if not (sys.version_info.major == 3 and sys.version_info.minor == 9):
    print("Project requires Python 3.9.*!")
    print(f"You are using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}.")
    sys.exit(1)

subprocess.check_call([sys.executable, "-m", "virtualenv", ".ikt590_virtualenv"])

if (sys.base_prefix == sys.prefix):
    subprocess.check_call([sys.executable, "-m", "virtualenv", ".ikt590_virtualenv"])
    sys.exit(1)
elif (sys.base_prefix != sys.prefix):
    subprocess.check_call([".ikt590_virtualenv\Scripts\python.exe", "-m", "pip", "install", "-r", "requirements.txt"])
    sys.exit(1)
else:
    print("Something has gone terribly wrong. You shouldn't see this error.")
