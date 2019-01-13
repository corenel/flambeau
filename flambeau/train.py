import signal
import sys

import click
import torch
import torch.multiprocessing as mp
from flambeau.misc import config, logger


@click.command(name='Training glow model')
@click.argument('profile', type=click.Path(exists=True))
@click.option('--data-root', type=click.Path(exists=True), default=None)
@click.option('--result-dir', type=click.Path(exists=True), default=None)
@click.option('--snapshot', type=click.Path(exists=True), default=None)
@click.option('--gpu', type=int, default=None)
def cli(profile, data_root, result_dir, snapshot, gpu):
    # load hyper-parameters
    hps = config.load_profile(profile, data_root, result_dir, snapshot,
                              gpu)


def main_worker(gpu, ngpus_per_node, hps):
    pass


if __name__ == '__main__':
    # this enables a Ctrl-C without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    # initialize logging
    logger.init_output_logging()

    # command
    cli()
