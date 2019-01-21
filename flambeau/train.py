import signal
import sys

import click

from flambeau.data import DatasetLoader
from flambeau.misc import config, logger
from flambeau.engine import BaseBuilder, BaseTrainer


@click.command(name='Training glow model')
@click.argument('profile', type=click.Path(exists=True))
@click.option('--data-root', type=click.Path(exists=True), default=None, help='root path of dataset')
@click.option('--result-dir', type=click.Path(exists=True), default=None, help='root path of result directory')
@click.option('--snapshot', type=click.Path(exists=True), default=None, help='path to snapshot file')
@click.option('--gpu', type=int, default=None, help='desired gpu id')
def cli(profile, data_root, result_dir, snapshot, gpu):
    # load hyper-parameters
    hps = config.load_profile(profile, data_root, result_dir, snapshot,
                              gpu)

    # load dataset
    loader = DatasetLoader(hps)
    dataset = loader.load()

    # build graph
    builder = BaseBuilder(hps)
    state = builder.build(training=True)
    trainer = BaseTrainer(hps=hps, dataset=dataset, **state)

    # train graph
    trainer.train()


if __name__ == '__main__':
    # this enables a Ctrl-C without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    # initialize logging
    logger.init_output_logging()

    # command
    cli()
