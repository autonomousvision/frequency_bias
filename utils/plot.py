import os
import torch
from math import sqrt
import matplotlib.pyplot as plt
from distutils.spawn import find_executable


HAS_LATEX = find_executable('latex')
if HAS_LATEX:        # use LaTeX fonts in plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=26)
else:
    plt.rc('font', family='serif', size=20)


def plot_std(mean, std, ax, x=None, **kwargs):
    if x is None:
        x = range(len(mean))

    l = ax.plot(x, mean, **kwargs)
    ax.fill_between(x, mean - std, mean + std, color=l[0]._color, alpha=0.3)


def plot_spectrum(spec_real, spec_gen, resolution, filename):
    fig, ax = plt.subplots(1)
    mean_real, std_real = spec_real['mean'][1:], spec_real['std'][1:]
    mean_gen, std_gen = spec_gen['mean'][1:], spec_gen['std'][1:]

    plot_std(mean_real, std_real, ax, c='C0', ls='--', label='ground truth')
    plot_std(mean_gen, std_gen, ax, c='orange', ls='-', label='prediction')

    # Settings for x-axis
    N = sqrt(2) * resolution
    fnyq = (N - 1) // 2
    x_ticks = [0, fnyq / 2, fnyq]
    x_ticklabels = ['%.1f' % (l / fnyq) for l in x_ticks]

    ax.set_xlim(0, fnyq)
    xlabel = r'$f/f_{nyq}$' if HAS_LATEX else 'f/fnyq'
    ax.set_xlabel(xlabel)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)

    # Settings for y-axis
    ax.set_ylabel(r'Spectral density')
    if std_gen.isfinite().all():
        ymin = (mean_real-std_real).min()
        if ymin < 0:
            ymin = mean_real.min()
        y_lim = ymin * 0.1, (mean_real+std_real).max() * 1.1
    else:
        y_lim = mean_real.min() * 0.1, mean_real.max() * 1.1
    ax.set_ylim(y_lim)
    ax.set_yscale('log')

    # Legend
    fs = plt.rcParams['font.size']
    plt.rcParams.update({'font.size': fs*0.75})
    ax.legend(loc='upper right', ncol=2, columnspacing=1)
    plt.rc('font', size=fs)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_spectrum_error_evolution(spec_real, spec_gen_all, resolution, filename):
    fig, ax = plt.subplots(1)

    # ensure spec_gen_all are in correct order
    spec_gen_all = sorted(spec_gen_all, key=lambda x: x['it'])

    # compute error image
    niter = spec_gen_all[-1]['it']
    nspec = len(spec_gen_all)
    iters = torch.linspace(0, niter, nspec).to(torch.long)
    mean_real = spec_real['mean'][1:]
    error_img = torch.empty(nspec, len(mean_real))
    for i, spec_gen in enumerate(spec_gen_all):
        mean_gen = spec_gen['mean'][1:]
        assert spec_gen['it'] == iters[i]
        error_img[i] = mean_gen / mean_real - 1

    # clamp at 100% relative error
    error_img.clamp_(-1, 1)

    # plot
    cmap = plt.cm.get_cmap('bwr')
    aspect = len(mean_real) / nspec                   # aspect=H/W, make image square
    h = ax.imshow(error_img, cmap=cmap, vmin=-1, vmax=1, origin='lower', aspect=aspect)

    # Settings for x-axis
    N = sqrt(2) * resolution
    fnyq = (N - 1) // 2
    x_ticks = [0, fnyq / 2, fnyq]
    x_ticklabels = ['%.1f' % (l / fnyq) for l in x_ticks]

    ax.set_xlim(0, fnyq)
    xlabel = r'$f/f_{nyq}$' if HAS_LATEX else 'f/fnyq'
    ax.set_xlabel(xlabel)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)

    # Settings for y-axis
    y_ticks = [0, nspec // 2, nspec]
    y_ticklabels = [t // 1000 for t in [0, niter // 2, niter]]

    ax.set_ylabel(r'Training Iteration [it/1000]')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)

    # Colorbar
    fig.colorbar(h, ticks=[-1, 0, 1], fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    from glob import glob
    import pickle
    spec_file_real = '../data/baboon/spectrum64_N1.pkl'
    traindir = '../output/generator_testbed/pggan'
    evaldir = os.path.join(traindir, 'eval')
    resolution = 64

    with open(spec_file_real, 'rb') as f:
        spec_real = pickle.load(f)

    spec_files_gen_all = glob(os.path.join(traindir, 'logs', 'spectrum_*.pkl'))
    spec_gen_all = []
    for path in spec_files_gen_all:
        with open(path, 'rb') as f:
            stats = pickle.load(f)
        spec_gen_all.append(stats)

    filename = os.path.join(evaldir, 'spectrum_error_evolution.png')
    plot_spectrum_error_evolution(spec_real, spec_gen_all, resolution, filename)