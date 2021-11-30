from models.pggan_generator import PGGANGenerator
from models.pggan_discriminator import PGGANDiscriminator

from models.direct_generator import DirectGenerator
from models.mlp import MLP

import os
if os.path.isfile('models/stylegan3/stylegan3/train.py'):
    from models.stylegan2_generator import SG2Generator
    from models.stylegan2_discriminator import SG2Discriminator
    from models.stylegan3_generator import SG3Generator
else:
    print('StyleGAN3 submodule not initialized. If you want to add StyleGAN3 support run: \n'
          '\tcd models/stylegan3 \n'
          '\tgit submodule update --init --recursive --remote')