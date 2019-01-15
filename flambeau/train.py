import signal
import sys

import click
import torch
import torch.multiprocessing as mp
from flambeau.misc import config, logger


@click.command(name='Training glow model')
@click.argument('profile', type=click.Path(exists=True))
@click.option('--data-root', type=click.Path(exists=True), default=None, help='root path of dataset')
@click.option('--result-dir', type=click.Path(exists=True), default=None, help='root path of result directory')
@click.option('--snapshot', type=click.Path(exists=True), default=None, help='path to snapshot file')
@click.option('--gpu', type=int, default=None, help='desired gpu id')
@click.option('--dist-url', type=str, default=None, help='url used to set up distributed training')
@click.option('--rank', type=int, default=None, help='node rank for distributed training')
def cli(profile, data_root, result_dir, snapshot, gpu, dist_url, rank):
    # load hyper-parameters
    hps = config.load_profile(profile, data_root, result_dir, snapshot,
                              gpu, dist_url, rank)

    ngpus_per_node = torch.cuda.device_count()
    if hps.device.distributed.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        hps.device.distributed.world_size = ngpus_per_node * hps.device.distributed.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, hps))
    else:
        # Simply call main_worker function
        main_worker(hps.device.graph[0], ngpus_per_node, hps)


def main_worker(gpu, ngpus_per_node, hps):
    pass


if __name__ == '__main__':
    # this enables a Ctrl-C without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    # initialize logging
    logger.init_output_logging()

    # command
    cli()
