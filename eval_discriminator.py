"""Train a Generator architecture with an MSE loss."""

import argparse
import os
import torch
from glob import glob
import pickle
from tqdm import tqdm
from PIL import Image
from utils import CheckpointIO
import utils.misc as misc
import utils.plot as plot
import utils.metrics as metrics
from dataset import get_dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt; plt.switch_backend('Agg'); plt.ioff()
    # Arguments
    parser = argparse.ArgumentParser(
        description='Evaluate image regression with a trained Generator.'
    )
    parser.add_argument('expname', type=str, help='Name of experiment.')
    parser.add_argument('--psnr', action='store_true', help='Evaluate PSNR of regressed images.')
    parser.add_argument('--image-evolution', action='store_true', help='Create video of image evolution.')
    parser.add_argument('--spectrum-evolution', action='store_true', help='Create video of spectrum evolution.')
    parser.add_argument('--spectrum-error-evolution', action='store_true', help='Create image of spectrum error evolution.')

    args = parser.parse_args()
    run_dir = os.path.join('output/discriminator_testbed', args.expname)
    cfg = misc.load_config(os.path.join(run_dir, 'config.yaml'))

    # fix random seed (ensures to sample same latent codes as in training)
    torch.manual_seed(cfg['training']['seed'])
    torch.cuda.manual_seed_all(cfg['training']['seed'])

    device = torch.device("cuda:0")

    # Short hands
    batch_size = cfg['training']['batch_size']
    nworkers = cfg['training']['nworkers']
    out_dir = os.path.join(run_dir, 'eval')
    log_dir = os.path.join(run_dir, 'logs')
    img_dir = os.path.join(run_dir, 'imgs')
    plot_dir = os.path.join(run_dir, 'plots')

    # Create missing directories
    os.makedirs(out_dir, exist_ok=True)

    if args.psnr or args.spectrum_error_evolution:               # Load dataset
        # Dataset
        dataset = get_dataset(cfg)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nworkers,
                                                 pin_memory=True, drop_last=False)

    if args.psnr:       # Load trained model to evaluate psnr of all images
        print('Evaluate PSNR...')
        # Logger
        checkpoint_io = CheckpointIO(checkpoint_dir=run_dir)

        # Create models
        common_kwargs = misc.EasyDict(resolution=cfg.data.resolution)
        generator = misc.construct_class_by_name(class_name="models.DirectGenerator", z=dataset.z, **common_kwargs)

        # Put generator on gpu if needed
        generator = generator.to(device)

        # Register modules to checkpoint
        checkpoint_io.register_modules(
            generator=generator,
        )

        # Load checkpoint
        load_dict = checkpoint_io.load('model.pt')
        print(f'Using checkpoint from iteration {load_dict["it"]}.')

        psnr = []
        for img, z in tqdm(dataloader):
            img, z = img.to(device), z.to(device)

            generator = generator.eval()
            pred = generator(z)
            psnr.append(metrics.psnr(pred, img))
        psnr = torch.cat(psnr).mean().item()

        print(f'Average PSNR: {psnr:.1f}.')

    if args.image_evolution:
        print('Plot image evolution...')
        images = [Image.open(f) for f in sorted(glob(os.path.join(img_dir, 'samples_*.png')))]
        misc.make_video(images, os.path.join(out_dir, 'image_evolution.mp4'), fps=20, quality=8)
        print('Done.')

    if args.spectrum_evolution:
        print('Plot spectrum evolution...')
        images = [Image.open(f) for f in sorted(glob(os.path.join(plot_dir, 'spectrum_*.png')))]
        misc.make_video(images, os.path.join(out_dir, 'spectrum_evolution.mp4'), fps=20, quality=8, macro_block_size=None)
        print('Done.')

    if args.spectrum_error_evolution:
        print('Plot spectrum error evolution...')
        spec_file_real = os.path.join(dataset.root, f'spectrum{dataset.resolution}_N{len(dataset)}.pkl')       # Loads cache file from training
        with open(spec_file_real, 'rb') as f:
            spec_real = pickle.load(f)

        spec_gen_all = []
        for spec_file in sorted(glob(os.path.join(log_dir, 'spectrum_*.pkl'))):
            with open(spec_file, 'rb') as f:
                spec = pickle.load(f)
            spec_gen_all.append(spec)

        filename = os.path.join(out_dir, 'spectrum_error_evolution.png')
        plot.plot_spectrum_error_evolution(spec_real, spec_gen_all, dataset.resolution, filename)
        print('Done.')
