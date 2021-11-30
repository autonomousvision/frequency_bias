import os
import pickle
import torch
from torch.fft import fftn
from math import sqrt, ceil


def resolution_to_spectrum_length(res):
    res_spec = (res+1) // 2
    res_azim_avg = ceil(sqrt(2) * res_spec)
    return res_azim_avg


def roll_quadrants(data):
    """
    Shift low frequencies to the center of fourier transform, i.e. [-N/2, ..., +N/2] -> [0, ..., N-1]
    Args:
        data: fourier transform, (NxHxW)

    Returns:
    Shifted fourier transform.
    """
    dim = data.ndim - 1

    if dim != 2:
        raise AttributeError(f'Data must be 2d but it is {dim}d.')
    if any(s % 2 == 0 for s in data.shape[1:]):
        raise RuntimeWarning('Roll quadrants for 2d input should only be used with uneven spatial sizes.')

    # for each dimension swap left and right half
    dims = tuple(range(1, dim + 1))  # add one for batch dimension
    shifts = torch.tensor(data.shape[1:]).div(2, rounding_mode='floor')  # N/2 if N even, (N-1)/2 if N odd
    return data.roll(shifts.tolist(), dims=dims)


def batch_fft(data, normalize=False):
    """
    Compute fourier transform of batch.
    Args:
        data: input tensor, (NxHxW)

    Returns:
    Batch fourier transform of input data.
    """

    dim = data.ndim - 1  # subtract one for batch dimension
    if dim != 2:
        raise AttributeError(f'Data must be 2d but it is {dim}d.')

    dims = tuple(range(1, dim + 1))  # add one for batch dimension

    if not torch.is_complex(data):
        data = torch.complex(data, torch.zeros_like(data))
    freq = fftn(data, dim=dims, norm='ortho' if normalize else 'backward')

    return freq


def azimuthal_average(image, center=None):
    # modified to tensor inputs from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    """
    Calculate the azimuthally averaged radial profile.
    Requires low frequencies to be at the center of the image.
    Args:
        image: Batch of 2D images, NxHxW
        center: The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    Returns:
    Azimuthal average over the image around the center
    """
    # Check input shapes
    assert center is None or (len(center) == 2), f'Center has to be None or len(center)=2 ' \
                                                 f'(but it is len(center)={len(center)}.'
    # Calculate the indices from the image
    H, W = image.shape[-2:]
    h, w = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))

    if center is None:
        center = torch.tensor([(w.max() - w.min()) / 2.0, (h.max() - h.min()) / 2.0])

    # Compute radius for each pixel wrt center
    r = torch.stack([w - center[0], h - center[1]]).norm(2, 0)

    # Get sorted radii
    r_sorted, ind = r.flatten().sort()
    i_sorted = image.flatten(-2, -1)[..., ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.long()  # attribute to the smaller integer

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented, computes bin change between subsequent radii
    rind = torch.where(deltar)[0]  # location of changed radius

    # compute number of elements in each bin
    nind = rind + 1  # number of elements = idx + 1
    nind = torch.cat([torch.tensor([0]), nind, torch.tensor([H * W])])  # add borders
    nr = nind[1:] - nind[:-1]  # number of radius bin, i.e. counter for bins belonging to each radius

    # Cumulative sum to figure out sums for each radius bin
    if H % 2 == 0:
        raise NotImplementedError('Not sure if implementation correct, please check')
        rind = torch.cat([torch.tensor([0]), rind, torch.tensor([H * W - 1])])  # add borders
    else:
        rind = torch.cat([rind, torch.tensor([H * W - 1])])  # add borders
    csim = i_sorted.cumsum(-1, dtype=torch.float64)  # integrate over all values with smaller radius
    tbin = csim[..., rind[1:]] - csim[..., rind[:-1]]
    # add mean
    tbin = torch.cat([csim[:, 0:1], tbin], 1)

    radial_prof = tbin / nr.to(tbin.device)  # normalize by counted bins

    return radial_prof


def get_spectrum(data, normalize=False):
    if (data.ndim - 1) != 2:
        raise AttributeError(f'Data must be 2d.')

    freq = batch_fft(data, normalize=normalize)
    power_spec = freq.real ** 2 + freq.imag ** 2
    N = data.shape[1]
    if N % 2 == 0:      # duplicate value for N/2 so it is put at the end of the spectrum and is not averaged with the mean value
        N_2 = N//2
        power_spec = torch.cat([power_spec[:, :N_2+1], power_spec[:, N_2:N_2+1], power_spec[:, N_2+1:]], dim=1)
        power_spec = torch.cat([power_spec[:, :, :N_2+1], power_spec[:, :, N_2:N_2+1], power_spec[:, :, N_2+1:]], dim=2)

    power_spec = roll_quadrants(power_spec)
    power_spec = azimuthal_average(power_spec)
    return power_spec


def compute_spectrum_stats_for_dataset(dataset, batch_size=32):
    # Try to lookup from cache.
    resolution = dataset[0][0].shape[1]
    cache_file = os.path.join(dataset.root, f'spectrum{resolution}_N{len(dataset)}.pkl')
    if dataset.highpass:
        cache_file = cache_file.replace('.pkl', '_highpass.pkl')
    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Main loop.
    spectra = []
    for data in torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False):
        imgs = data[0]
        if imgs.shape[1] == 1:
            imgs = imgs.repeat([1, 3, 1, 1])
        imgs = imgs.to('cuda:0', torch.float32)
        spec = get_spectrum(imgs.flatten(0, 1)).unflatten(0, (imgs.shape[0], imgs.shape[1]))
        spec = spec.mean(dim=1)     # average over channels
        spectra.append(spec.cpu())

    spectra = torch.cat(spectra)
    stats = {'mean': spectra.mean(dim=0), 'std': spectra.std(dim=0)}

    # Save to cache.
    with open(cache_file, 'wb') as f:
        pickle.dump(stats, f)
    return stats


def compute_spectrum_stats_for_generator(dataset, model, batch_size=32):
    device = 'cuda:0'
    model = model.eval().to(device)

    # Main loop.
    spectra = []
    with torch.no_grad():
        for data in torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False):
            z = data[1].to(device)
            imgs = model(z)
            if imgs.shape[1] == 1:
                imgs = imgs.repeat([1, 3, 1, 1])
            imgs = imgs.to(torch.float32)
            spec = get_spectrum(imgs.flatten(0, 1)).unflatten(0, (imgs.shape[0], imgs.shape[1]))
            spec = spec.mean(dim=1)  # average over channels
            spectra.append(spec.cpu())

    spectra = torch.cat(spectra)
    stats = {'mean': spectra.mean(dim=0), 'std': spectra.std(dim=0)}
    return stats


def evaluate_spectrum(dataset, model, batch_size=32):
    spec_real = compute_spectrum_stats_for_dataset(dataset, batch_size=batch_size)
    spec_gen = compute_spectrum_stats_for_generator(dataset, model, batch_size=batch_size)

    return spec_real, spec_gen