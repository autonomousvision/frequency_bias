import sys
import os
from pathlib import Path

current_path = os.getcwd()

module_path = Path(__file__).parent / 'stylegan3'
sys.path.insert(0, str(module_path.resolve()))
os.chdir(module_path)

from training.networks_stylegan2 import Generator as SG2Generator, Discriminator as SG2Discriminator
from training.networks_stylegan3 import Generator as SG3Generator

os.chdir(current_path)
sys.path.pop(0)