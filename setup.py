# Automate setup process for setting up directories
import os

wd = os.getcwd()

os.makedirs(wd + '/data', exist_ok = True)

data_dir = wd + '/data'
dirs = ['/audio', '/model', '/temp', '/transcriptions', '/video-comp', '/video-orig', '/video-roi']

for dir in dirs:
    os.makedirs(dir, exist_ok = True)