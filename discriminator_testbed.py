"""Train a Discriminator architecture with a GAN loss."""

import argparse
import os
import torch
import pickle
import copy
from utils import CheckpointIO, Logger
import utils.misc as misc
import utils.plot as plot
import utils.spectrum as spectrum
import utils.gan_training as gan_training
from dataset import get_dataset
from loss import get_criterion
from torchvision.utils import save_image


if __name__ == '__main__':
    import matplotlib.pyplot as plt; plt.switch_backend('Agg'); plt.ioff()
    # Arguments
    parser = argparse.ArgumentParser(
        description='Image regression with a Discriminator.'
    )
    parser.add_argument('expname', type=str, help='Name of experiment.')
    parser.add_argument('config', type=str, help='Path to config file.')

    args = parser.parse_args()
    cfg = misc.load_config(args.config, 'configs/discriminator_testbed/default.yaml')

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
    out_dir = os.path.join('output/discriminator_testbed', args.expname)
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
    use_spec_disc = cfg.model.pop('spectrum_disc', False)
    common_kwargs = misc.EasyDict(resolution=cfg.data.resolution)
    generator = misc.construct_class_by_name(class_name="models.DirectGenerator", z=dataset.z,  **common_kwargs)
    generator_test = copy.deepcopy(generator)
    model = misc.construct_class_by_name(**cfg.model, label_size=len(dataset)-1,        # class conditional
                                         **common_kwargs).train().requires_grad_(True).to(device)
    if use_spec_disc:       # additional discriminator on log of reduced spectrum
        from models import MLP
        from functools import partial
        spec_len = spectrum.resolution_to_spectrum_length(cfg.data.resolution)
        model.spec_disc = MLP(input_size=spec_len,
                              output_size=len(dataset),
                              nhidden=0, dhidden=spec_len, activation=partial(torch.nn.LeakyReLU, negative_slope=0.2))
    print(generator)
    try:
        print(model)
    except TypeError:           # print model does not work with SG2 Discriminator
        pass
    print(f'Generator has {misc.count_trainable_parameters(generator)} trainable parameters.')
    print(f'Discriminator has {misc.count_trainable_parameters(model)} trainable parameters.')

    # Put model on gpu if needed
    generator = generator.to(device)
    generator_test = generator_test.to(device)
    model = model.to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.training.lr_g, betas=(0., 0.99), eps=1e-8)
    d_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr_d, betas=(0., 0.99), eps=1e-8)
    criterion = get_criterion(**cfg.training.criterion)

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        generator=generator,
        generator_test=generator_test,
        model=model,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
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

    def spec_disc_step(x, c=None):
        specs = spectrum.get_spectrum(x.flatten(0, 1), normalize=True).unflatten(0, x.shape[:2]).to(torch.float32)
        specs = specs.mean(dim=1)       # average over channels
        specs = (1 + specs).log()       # apply to logarithm of spectrum to avoid very large values
        return model.spec_disc(specs, c=c)

    def discriminator_trainstep(x_real, z, reg_param=10):
        gan_training.toggle_grad(generator, False)
        gan_training.toggle_grad(model, True)
        generator.train()
        model.train()
        d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        d_real = model(x_real, c=z)
        targets = d_real.new_full(size=d_real.size(), fill_value=1)
        if use_spec_disc:
            d_real_spec = spec_disc_step(x_real, c=z)
            d_real = torch.cat([d_real, d_real_spec])
            targets = targets.repeat(2, 1)

        dloss_real = criterion(d_real, targets)

        # Regularization on real
        dloss_real.backward(retain_graph=True)
        reg = reg_param * gan_training.compute_grad2(d_real, x_real).mean()
        reg.backward()

        # On fake data
        with torch.no_grad():
            x_fake = generator(z)

        x_fake.requires_grad_()

        d_fake = model(x_fake, c=z)
        targets = d_fake.new_full(size=d_fake.size(), fill_value=0)
        if use_spec_disc:
            d_fake_spec = spec_disc_step(x_fake, c=z)
            d_fake = torch.cat([d_fake, d_fake_spec])
            targets = targets.repeat(2, 1)

        dloss_fake = criterion(d_fake, targets)
        dloss_fake.backward()

        d_optimizer.step()

        gan_training.toggle_grad(model, False)

        # Output
        dloss = (dloss_real + dloss_fake)
        return dloss.item(), reg.item()

    def generator_trainstep(z):
        gan_training.toggle_grad(generator, True)
        gan_training.toggle_grad(model, False)
        generator.train()
        model.train()
        g_optimizer.zero_grad()

        x_fake = generator(z)
        d_fake = model(x_fake, c=z)
        targets = d_fake.new_full(size=d_fake.size(), fill_value=1)
        if use_spec_disc:
            d_fake_spec = spec_disc_step(x_fake, c=z)
            d_fake = torch.cat([d_fake, d_fake_spec])
            targets = targets.repeat(2, 1)

        gloss = criterion(d_fake, targets)
        gloss.backward()

        g_optimizer.step()
        return gloss.item()

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
                dloss, reg = discriminator_trainstep(img, z, reg_param=cfg.training.reg_param)
                logger.add('losses', 'discriminator', dloss, it=it)
                logger.add('losses', 'regularizer', reg, it=it)

                # Image updates
                gloss = generator_trainstep(z)
                logger.add('losses', 'generator', gloss, it=it)

                # Update ema
                gan_training.update_average(generator_test, generator, beta=cfg.training.model_average_beta)

                # Print stats
                if (it % cfg['training']['print_every']) == 0:
                    dloss_last = logger.get_last('losses', 'discriminator')
                    gloss_last = logger.get_last('losses', 'generator')
                    print('[epoch %0d, it %4d] dloss = %.4f, gloss = %.4f' % (epoch_idx, it, dloss_last, gloss_last))

            # Evaluate if necessary
            if (it % eval_every) == 0:
                generator_test.eval()
                # Evaluate spectrum
                spec_real, spec_gen = spectrum.evaluate_spectrum(dataset, generator_test, batch_size=batch_size)
                spec_gen.update({'it': it})
                filename = os.path.join(log_dir, f'spectrum_%08d.pkl' % it)
                with open(filename, 'wb') as f:
                    pickle.dump(spec_gen, f)

                # Save plot of spectrum
                filename = os.path.join(plot_dir, f'spectrum_%08d.png' % it)
                plot.plot_spectrum(spec_real, spec_gen, cfg.data.resolution, filename)

                # Save some generated images
                pred = generator_test(eval_z)
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
