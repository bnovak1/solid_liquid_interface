import numpy as np
import glob
import subprocess


files = glob.glob('test_*.*')
suffixes = [file.split('test_')[-1] for file in files]


for suffix in suffixes:
    subprocess.call('mv test_' + suffix + ' tests/results_TiNi/test_r_' + suffix, shell=True)
