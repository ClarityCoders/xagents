import os
import subprocess
import sys

if sys.platform == 'darwin':
    subprocess.run(['xhost', '+'])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
