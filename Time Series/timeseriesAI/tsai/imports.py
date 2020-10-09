import fastai2
from fastai2.imports import *
from fastai2.data.all import *
from fastai2.torch_core import *
from fastai2.learner import *
from fastai2.metrics import *
from fastai2.callback.all import *
from fastai2.vision.data import *
from fastai2.interpret import *
from fastai2.optimizer import *
from fastai2.torch_core import Module
from fastai2.data.transforms import get_files
import fastcore
from fastcore.test import *
from fastcore.utils import *
import torch
import torch.nn as nn
import psutil
import scipy as sp
import sklearn.metrics as skm
import gc
import os
from numbers import Integral
from pathlib import Path
import time
from time import gmtime, strftime
from IPython.display import Audio, display, HTML, Javascript
import tsai

PATH = Path(os.getcwd())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpus = defaults.cpus

def save_nb(verbose=False):
    display(Javascript('IPython.notebook.save_checkpoint()'))
    time.sleep(1)
    pv('\nCurrent notebook saved.\n', verbose)

def last_saved(max_elapsed=10):
    print('\n')
    lib_path = Path(os.getcwd()).parent
    folder = lib_path/'tsai'
    print('Checking folder:', folder)
    counter = 0
    elapsed = 0
    current_time = time.time()
    for fp in get_files(folder):
        fp = str(fp)
        fn = fp.split('/')[-1]
        if not fn.endswith(".py") or fn.startswith("_") or fn.startswith(".") or fn in ['imports.py', 'all.py']: continue
        elapsed_time = current_time - os.path.getmtime(fp)
        if elapsed_time > max_elapsed: 
            print(f"{fn:30} saved {elapsed_time:10.0f} s ago ***")
            counter += 1
        elapsed += elapsed_time
    if counter == 0: 
        print('Correct conversion! 😃')
        output = 1
    else: 
        print('Incorrect conversion! 😔')
        output = 0
    print(f'Total elapsed time {elapsed:.0f} s')
    print(strftime("%a, %d %b %Y %H:%M:%S %Z\n"),"\n")
    return output
    
def beep(inp=1):
    mult = 1.6*inp if inp else .08
    wave = np.sin(mult*np.arange(1000))
    return Audio(wave, rate=10000, autoplay=True)

def create_scripts(max_elapsed=10):
    from nbdev.export import notebook2script
    save_nb()
    notebook2script()
    return last_saved(max_elapsed)
    