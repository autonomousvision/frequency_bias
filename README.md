# Frequency Bias of Generative Models

![](gfx/teaser_gen.gif) | ![](gfx/teaser_disc.gif)
:---:| :---: 
Generator Testbed | Discriminator Testbed

This repository contains official code for the paper
[On the Frequency Bias of Generative Models](http://cvlibs.net/publications/Schwarz2021NEURIPS.pdf).

You can find detailed usage instructions for analyzing standard GAN-architectures and your own models below.


If you find our code or paper useful, please consider citing

    @inproceedings{Schwarz2021NEURIPS,
      title = {On the Frequency Bias of Generative Models},
      author = {Schwarz, Katja and Liao, Yiyi and Geiger, Andreas},
      booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
      year = {2021}
    }

## Installation
Please note, that this repo requires one GPU for running.
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `fbias` using
```
conda env create -f environment.yml
conda activate fbias
```

## Generator Testbed

You can run a demo of our generator testbed via:
```
chmod +x ./scripts/demo_generator_testbed.sh
./scripts/demo_generator_testbed.sh
```
This will train the Generator of [Progressive Growing GAN](https://arxiv.org/abs/1710.10196) to regress a single image.
Further, the training progression on the image regression, spectrum, and spectrum error are summarized in `output/generator_testbed/baboon64/pggan/eval`.

In general, to analyze the spectral properties of a generator architecture you can train a model by running
```
python generator_testbed.py *EXPERIMENT_NAME* *PATH/TO/CONFIG*
```
This script should create a folder `output/generator_testbed/*EXPERIMENT_NAME*` where you can find the training progress.
To evaluate the spectral properties of the trained model run
```
python eval_generator.py *EXPERIMENT_NAME* --psnr --image-evolution --spectrum-evolution --spectrum-error-evolution
```
This will print the average PSNR of the regressed images and visualize image evolution, spectrum evolution, 
and spectrum error evolution in `output/generator_testbed/*EXPERIMENT_NAME*/eval`.

## Discriminator Testbed

You can run a demo of our discriminator testbed via:
```
chmod +x ./scripts/demo_discriminator_testbed.sh
./scripts/demo_discriminator_testbed.sh
```
This will train the Discriminator of [Progressive Growing GAN](https://arxiv.org/abs/1710.10196) to regress a single image.
Further, the training progression on the image regression, spectrum, and spectrum error are summarized in `output/discriminator_testbed/baboon64/pggan/eval`.

In general, to analyze the spectral properties of a discriminator architecture you can train a model by running
```
python discriminator_testbed.py *EXPERIMENT_NAME* *PATH/TO/CONFIG*
```
This script should create a folder `output/discriminator_testbed/*EXPERIMENT_NAME*` where you can find the training progress.
To evaluate the spectral properties of the trained model run
```
python eval_discriminator.py *EXPERIMENT_NAME* --psnr --image-evolution --spectrum-evolution --spectrum-error-evolution
```
This will print the average PSNR of the regressed images and visualize image evolution, spectrum evolution, 
and spectrum error evolution in `output/discriminator_testbed/*EXPERIMENT_NAME*/eval`.


## Datasets

### Toyset

You can generate a toy dataset with Gaussian peaks as spectrum by running
```
cd data
python toyset.py 64 100
cd ..
```
This creates a folder `data/toyset/` and generates 100 images of resolution 64x64 pixels.

### CelebA-HQ

Download [celebA_hq](https://github.com/tkarras/progressive_growing_of_gans).
Then, update `data:root: *PATH/TO/CELEBA_HQ*` in the config file.

### Other datasets

The config setting `data:root: *PATH/TO/DATA*` needs to point to a folder with the training images.
You can use any dataset which follows the folder structure
```
*PATH/TO/DATA*/xxx.png
*PATH/TO/DATA*/xxy.png
...
```
By default, the images are center-cropped and optionally resized to the resolution specified in the config file under`data:resolution`.
Note, that you can also use a subset of images via `data:subset`.

## Architectures

### StyleGAN Support

In addition to [Progressive Growing GAN](https://arxiv.org/abs/1710.10196), this repository supports analyzing the following architectures
- [StyleGAN2](https://arxiv.org/abs/1912.04958) Generator
- [StyleGAN2](https://arxiv.org/abs/1912.04958) Discriminator
- [StyleGAN3](https://nvlabs-fi-cdn.nvidia.com/stylegan3/stylegan3-paper.pdf) Generator

For this, you need to initialize the stylegan3 submodule by running
```
git pull --recurse-submodules
cd models/stylegan3/stylegan3
git submodule init
git submodule update
cd ../../../
```

Next, you need to install any additional requirements for this repo. You can do this by running 
```
conda activate fbias
conda env update --file environment_sg3.yml --prune
```

You can now analyze the spectral properties of the StyleGAN architectures by running
```
# StyleGAN2
python generator_testbed.py baboon64/StyleGAN2 configs/generator_testbed/sg2.yaml
python discriminator_testbed.py baboon64/StyleGAN2 configs/discriminator_testbed/sg2.yaml
# StyleGAN3
python generator_testbed.py baboon64/StyleGAN3 configs/generator_testbed/sg3.yaml
```

### Other architectures

To analyze any other network architectures, you can add the respective model file (or submodule) under `models`.
You then need to write a wrapper class to integrate the architecture seamlessly into this code base. 
Examples for wrapper classes are given in
- `models/stylegan2_generator.py` for the Generator
- `models/stylegan2_discriminator.py` for the Discriminator


## Further Information

This repository builds on Lars Mescheder's awesome framework for [GAN training](https://github.com/LMescheder/GAN_stability).
Further, we utilize code from the [Stylegan3-repo](https://github.com/NVlabs/stylegan3.git) and [GenForce](https://github.com/genforce/genforce).