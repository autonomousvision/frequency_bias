import argparse
import torch
from tqdm import tqdm
from math import sqrt, pi
from torch.fft import irfftn
from torchvision.transforms import CenterCrop, ToPILImage


def gaussian(x, mu, sigma):
    return 1 / (sqrt(2 * pi) * sigma) * torch.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


def make_circ_magnitude(resolution, freq):
    """Create circular magnitude image using a 2D Gaussian with mean=freq and sigma~1pixel."""
    if resolution % 2 != 0:
        raise NotImplementedError

    # magnitude image of (real) image has shape: (res x res//2+1)
    # we first only create the first quadrant of shape (res//2 x res//2)
    spectrum_size = resolution // 2

    # Use a 2D gaussian to create circular spectrum
    r = torch.stack(
        torch.meshgrid(torch.linspace(0, 1, spectrum_size),
                       torch.linspace(0, 1, spectrum_size + 1),     # add one to store the mean value
                       )
    ).norm(dim=0)

    mean = freq
    std = 1 / (spectrum_size + 1)  # sigma = 1pxl
    magnitude = gaussian(r, mean, std)

    # We need to decide the normalization of the Gaussian
    # We choose it, s.t.
    #   image.norm() = fft(image, norm='ortho').norm() ~ resolution
    #   fft(image, norm='ortho').norm() = sqrt(magnitude.sum())
    #   -> magnitude.sum() ~ resolution**2
    # Since we consider only a single channel but image will have three channels, we get
    #   magnitude.sum() ~ resolution**2 / 3

    # Compute sum of a Gaussian with mean=1/sqrt(2) (middle of spectrum) and sigma~1pixel
    # to get the normalization constant
    norm = gaussian(r, 1 / sqrt(2), std).sum()
    norm *= 4               # we only considered one quadrant

    magnitude /= norm                           # normalize sum approx to 1
    magnitude *= resolution**2 / 3              # scale sum to approx resolution**2 / 3

    # Stack to get the full magnitude image of shape (res x res//2+1)
    magnitude = torch.cat([magnitude, magnitude.flipud()])
    return magnitude


def img_from_magnitude(magnitude):
    """Combine a given spectrum (magnitude) with a random phase and use inverse Fourier transform to obtain an image."""
    magnitude = magnitude.repeat(3, 1, 1)           # create 3 channels

    phase = torch.rand_like(magnitude) * 2 * pi  # uniform distributed phase
    # set phase of mean to zero
    phase[0, 0] = 0
    phase[0, -1] = 0
    phase[magnitude.shape[0]//2, 0] = 0
    phase[magnitude.shape[0]//2, -1] = 0

    tanphi = torch.tan(phase)

    # compute real and imaginary part
    # tan(phase) = Im/Re -> Im = Re*tan(phase)
    # mag^2 = Im^2 + Re^2 -> mag^2 = (1+tan(phase)^2)*Re^2 -> Re = mag/sqrt(1+tan(phase)^2)
    real = magnitude / (1 + tanphi ** 2).sqrt()
    imag = real * tanphi

    freq_img = torch.complex(real, imag)
    img = irfftn(freq_img, dim=(1, 2), norm='ortho')

    return img


def generate_toysamples(resolution, freqrange=[(0.05, 0.15), (0.75, 0.85)]):
    assert isinstance(freqrange, list) and isinstance(freqrange[0], tuple)
    resolution_gen = 2*resolution          # generate images at higher resolution to avoid discretization artifacts
    crop = CenterCrop((resolution, resolution))     # center crop generated images <-> downsample spectrum
    to_pil = ToPILImage()

    fnyq = sqrt(2)          # diagonal of image in range [0,1]
    while True:
        img = torch.zeros(3, resolution_gen, resolution_gen)
        for fmin, fmax in freqrange:
            f = torch.rand(1) * (fmax - fmin) + fmin
            f = f * fnyq

            img += img_from_magnitude(make_circ_magnitude(resolution_gen, f))

        img /= len(freqrange)           # compute mean
        # convert to uint8, assume range is [-1, 1]
        img = ((img + 1) * 127.5).clamp_(0, 255).to(torch.uint8)
        img = to_pil(crop(img))
        yield img


if __name__ == '__main__':
    import os
    import sys
    sys.path.append('..')
    # Arguments
    parser = argparse.ArgumentParser(
        description='Generate toyimages with multiple Gaussian peaks as spectrum.'
    )
    parser.add_argument('res', type=int, help='Image resolution.')
    parser.add_argument('nsamples', type=int, help='Number of samples to generate.')
    parser.add_argument('--outdir', type=str, help='Directory for saving the images.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--plot_stats', action='store_true', help='Plot spectrum statistics of generated toyset.')
    parser.add_argument('--freqs', type=str, nargs='+',
                        default=['0.05,0.15', '0.75, 0.85'],
                        help='List of frequency ranges (fmin,fmax) wrt. the nyquist frequency.'
                             'The number of freqranges equals the number of peaks in the spectrum of each image. '
                             'The mean value of each peak i is drawn uniformly from (fmin,fmax)_i.')

    args = parser.parse_args()
    try:
        args.freqs = [tuple(map(float, s.split(','))) for s in args.freqs]
        for f in args.freqs:
            if len(f) != 2:
                raise TypeError
    except:
        raise TypeError("freqranges must be fmin,fmax")

    if args.outdir is None:
        args.outdir = f'toyset{args.res}_{args.nsamples}'
    os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(args.seed)
    i = 0
    for img in tqdm(generate_toysamples(resolution=args.res), total=args.nsamples, desc='Creating toyset'):
        img.save(os.path.join(args.outdir, '%08d.png' % i))
        i += 1
        if i == args.nsamples:
            break

    if args.plot_stats:
        import matplotlib.pyplot as plt; plt.switch_backend('TkAgg'); plt.ion()
        from glob import glob
        from PIL import Image
        from torchvision.transforms import ToTensor
        from utils.spectrum import get_spectrum
        from utils.plot import plot_std, HAS_LATEX

        imgs = torch.stack([ToTensor()(Image.open(f)) for f in glob(os.path.join(args.outdir, '*.png'))])
        imgs = imgs * 2 - 1
        spectra = get_spectrum(imgs.flatten(0, 1)).unflatten(0, (args.nsamples, 3)).mean(dim=1)

        fig, ax = plt.subplots(1)

        # Settings for x-axis
        N = sqrt(2) * args.res
        fnyq = (N - 1) / 2
        x_ticks = [0, fnyq / 2, fnyq]
        x_ticklabels = ['%.1f' % (l / fnyq) for l in x_ticks]

        ax.set_xlim(0, fnyq)
        xlabel = r'$f/f_{nyq}$' if HAS_LATEX else 'f/fnyq'
        ax.set_xlabel(xlabel)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels)

        # Settings for y-axis
        ax.set_ylabel(r'Spectral density')
        ax.set_yscale('log')

        plot_std(spectra.mean(dim=0), spectra.std(dim=0), ax=ax)
        for freqs in args.freqs:
            ax.axvline(sum(freqs) / 2 * fnyq, c='k', ls='--')

        plt.show()
        plt.waitforbuttonpress()