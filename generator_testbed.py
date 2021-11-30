"""Train a Generator architecture with an MSE loss."""

import argparse
import os
import torch
import pickle
from utils import CheckpointIO, Logger
import utils.misc as misc
import utils.plot as plot
import utils.spectrum as spectrum
from dataset import get_dataset
from loss import get_criterion
from torchvision.utils import save_image


if __name__ == '__main__':
    import matplotlib.pyplot as plt; plt.switch_backend('Agg'); plt.ioff()
    # Arguments
    parser = argparse.ArgumentParser(
        description='Image regression with a Generator.'
    )
    parser.add_argument('expname', type=str, help='Name of experiment.')
    parser.add_argument('config', type=str, help='Path to config file.')

    args = parser.parse_args()
    cfg = misc.load_config(args.config, 'configs/generator_testbed/default.yaml')

    # fix random seed
    torch.manual_seed(cfg['training']['seed'])
    torch.cuda.manual_seed_all(cfg['training']['seed'])

    device = torch.device("cuda:0")

    # Short hands
    batch_size = cfg['training']['batch_size']
    nworkers = cfg['training']['nworkers']
    nepochs = cfg['training']['nepochs']
    eval_every = cfg['training']['eval_every']
    save_every = cfg['training']['save_every']
    out_dir = os.path.join('output/generator_testbed', args.expname)
    log_dir = os.path.join(out_dir, 'logs')
    img_dir = os.path.join(out_dir, 'imgs')
    plot_dir = os.path.join(out_dir, 'plots')

    # Create missing directories
    for d in [log_dir, img_dir, plot_dir]:
        os.makedirs(d, exist_ok=True)

    # Logger
    checkpoint_io = CheckpointIO(checkpoint_dir=out_dir)

    # Save config
    misc.save_config(os.path.join(out_dir, 'config.yaml'), cfg)

    # Dataset
    dataset = get_dataset(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nworkers,
                                             pin_memory=True, drop_last=False)

    # Visualize training data
    grid_size = (8, 4)
    images = [dataset[i][0] for i in range(min(len(dataset), grid_size[0]*grid_size[1]))]
    eval_z = torch.stack([dataset[i][1] for i in range(min(len(dataset), grid_size[0]*grid_size[1]))]).to(device)
    save_image(images, os.path.join(out_dir, 'training_data.png'), nrow=grid_size[1], normalize=True, value_range=(-1, 1))

    # Create models
    common_kwargs = misc.EasyDict(resolution=cfg.data.resolution)
    model = misc.construct_class_by_name(**cfg.model, **common_kwargs).train().requires_grad_(True).to(device)
    print(model)
    print(f'Model has {misc.count_trainable_parameters(model)} trainable parameters.')

    # Put model on gpu if needed
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, betas=(0., 0.99), eps=1e-8)
    criterion = get_criterion(**cfg.training.criterion)

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        model=model,
        optimizer=optimizer,
    )

    # Logger
    logger = Logger(
        log_dir=log_dir,
        img_dir=img_dir,
        monitoring=cfg['training']['monitoring'],
        monitoring_dir=os.path.join(out_dir, 'monitoring')
    )

    # Load checkpoint if it exists
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileNotFoundError:
        epoch_idx = -1
        it = -1
    else:
        epoch_idx = load_dict.get('epoch_idx')
        it = load_dict.get('it')
        logger.load_stats('stats.p')

    # Training loop
    print('Start training...')
    while epoch_idx < nepochs:
        epoch_idx += 1

        for img, z in dataloader:
            it += 1
            img, z = img.to(device), z.to(device)

            if it > 0:      # only evaluate at initialization
                model = model.train()

                # Model updates
                optimizer.zero_grad()
                pred = model(z)
                loss = criterion(pred, img)
                loss.backward()

                optimizer.step()

                logger.add('losses', 'train', loss, it=it)

                # Print stats
                if (it % cfg['training']['print_every']) == 0:
                    loss_last = logger.get_last('losses', 'train')
                    print('[epoch %0d, it %4d] loss = %.4f' % (epoch_idx, it, loss_last))

            # Evaluate if necessary
            if (it % eval_every) == 0:
                model = model.eval()
                # Evaluate spectrum
                spec_real, spec_gen = spectrum.evaluate_spectrum(dataset, model, batch_size=batch_size)
                spec_gen.update({'it': it})
                filename = os.path.join(log_dir, f'spectrum_%08d.pkl' % it)
                with open(filename, 'wb') as f:
                    pickle.dump(spec_gen, f)

                # Save plot of spectrum
                filename = os.path.join(plot_dir, f'spectrum_%08d.png' % it)
                plot.plot_spectrum(spec_real, spec_gen, cfg.data.resolution, filename)

                # Save some generated images
                pred = model(eval_z)
                filename = os.path.join(img_dir, 'samples_%08d.png' % it)
                save_image(pred, filename, normalize=True, value_range=(-1, 1))

        # (iii) Checkpoint if necessary
        if (epoch_idx % save_every) == 0 and (it > 0):
            print('Saving checkpoint...')
            checkpoint_io.save('model.pt', epoch_idx=epoch_idx, it=it)
            logger.save_stats('stats.p')

    # Save model
    print('Saving last model...')
    checkpoint_io.save('model.pt', epoch_idx=epoch_idx, it=it)
    logger.save_stats('stats.p')
