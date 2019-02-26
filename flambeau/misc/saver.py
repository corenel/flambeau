import glob
import json
import os
import re
import shutil

import torch

from .logger import set_output_log_file
from .config import dump_profile


def create_result_subdir(result_dir, desc, profile, copy=False):
    """
    Create and initialize result sub-directory

    :param result_dir: path to root of result directory
    :type result_dir: str
    :param desc: description of current experiment
    :type desc: str
    :param profile: profile
    :param copy: whether or not to copy source code to the result subdir
    :type copy: bool
    :return: path to result sub-directory
    :rtype: str
    """
    # determine run id
    run_id = 0
    for fname in glob.glob(os.path.join(result_dir, '*')):
        fbase = os.path.basename(fname)
        finds = re.findall(r'^([\d]+)-', fbase)
        if len(finds) != 0:
            ford = int(finds[0])
            run_id = max(run_id, ford + 1)

    # create result sub-directory
    result_subdir = os.path.join(result_dir, '{:03d}-{:s}'.format(
        run_id, desc))
    if not os.path.exists(result_subdir):
        os.makedirs(result_subdir)
    set_output_log_file(os.path.join(result_subdir, 'log.txt'))
    print('[Builder] Saving results to {}'.format(result_subdir))

    # export profile
    dump_profile(profile, result_subdir, file_type='yml')

    # copy source code
    if copy:
        src_path = os.getcwd()
        dst_path = os.path.join(result_subdir, 'code')
        print('[Builder] Copying {} to {}'.format(src_path, dst_path))
        # shutil.copytree(os.getcwd(), os.path.join(result_subdir, 'code'))
        os.system('rsync --filter=":- .gitignore" -azqP "{}" "{}"'.format(
            src_path, dst_path))

    return result_subdir


def locate_result_subdir(result_dir, run_id_or_result_subdir):
    """
    Locate result subdir by given run id or path

    :param result_dir: path to root of result directory
    :type result_dir: str
    :param run_id_or_result_subdir: run id or subdir path
    :type run_id_or_result_subdir: int or str
    :return: located result subdir
    :rtype: str
    """
    if isinstance(run_id_or_result_subdir,
                  str) and os.path.isdir(run_id_or_result_subdir):
        return run_id_or_result_subdir

    searchdirs = ['', 'results', 'networks']

    for searchdir in searchdirs:
        d = result_dir if searchdir == '' else os.path.join(
            result_dir, searchdir)
        # search directly by name
        d = os.path.join(d, str(run_id_or_result_subdir))
        if os.path.isdir(d):
            return d
        # search by prefix
        if isinstance(run_id_or_result_subdir, int):
            prefix = '{:03d}'.format(run_id_or_result_subdir)
        else:
            prefix = str(run_id_or_result_subdir)
        dirs = sorted(
            glob.glob(os.path.join(result_dir, searchdir, prefix + '-*')))
        dirs = [d for d in dirs if os.path.isdir(d)]
        if len(dirs) == 1:
            return dirs[0]
    print('[Builder] Cannot locate result subdir for run: {}'.format(
        run_id_or_result_subdir))
    return None


def get_model_name(epoch):
    """
    Return filename of model snapshot by epoch

    :param epoch: global epoch of model
    :type epoch: int
    :return: model snapshot file name
    :rtype: str
    """
    return 'network-snapshot-{:04d}.pth'.format(epoch)


def get_best_model_name():
    """
    Return filename of best model snapshot by step

    :return: filename of best model snapshot
    :rtype: str
    """
    return 'network-snapshot-best.pth'


def get_latest_model_name(result_subdir):
    """
    Return filename of best model snapshot by step

    :param result_subdir: path to result sub-directory
    :type result_subdir: str
    :return: filename of last model snapshot
    :rtype: str
    """
    latest = -1
    for f in os.listdir(result_subdir):
        if os.path.isfile(os.path.join(result_subdir, f)) and \
                re.search(r'network-snapshot-([\d]+).pth', f):
            f_step = int(re.findall(r'network-snapshot-([\d]+).pth', f)[0])
            if latest < f_step:
                latest = f_step

    if latest == -1:
        assert FileNotFoundError('No latest model in {}'.format(result_subdir))

    return get_model_name(latest)


def save_snapshot(graph,
                  epoch,
                  result_subdir,
                  model_name=None,
                  is_best=False,
                  state=None):
    """
    Save snapshot

    :param graph: model
    :type graph: torch.nn.Module
    :param epoch: epoch index
    :type epoch: int
    :param result_subdir: path to save
    :type result_subdir: str
    :param model_name: model name
    :type model_name: str
    :param is_best: whether or not is best model
    :type is_best: bool
    :param state: other state to save
    :type state: dict
    """
    state_to_save = {
        'graph': graph.module.state_dict() if hasattr(graph, 'module') else graph.state_dict(),
        'epoch': epoch
    }
    if state is not None:
        state_to_save.update(state)

    # save current state
    if model_name is None:
        model_name = get_model_name(epoch)
    save_path = os.path.join(result_subdir, model_name)
    torch.save(state_to_save, save_path)

    # save best state
    if is_best:
        best_path = os.path.join(result_subdir, get_best_model_name())
        shutil.copy(save_path, best_path)


def load_snapshot(model_path):
    """
    Load snapshot

    :param model_path: path to snapshot
    :type model_path: str
    :return: built state
    :rtype: dict
    """
    state = torch.load(model_path)
    return state


class Saver:

    def __init__(self, hps, training=True):
        super().__init__()
        self.hps = hps
        self.training = training
        self.result_dir = hps.general.result_dir
        self.result_subdir = self._init_subdir()

    def load(self, graph):
        state = None
        if self.hps.general.warm_start:
            step_or_model_path = None
            if os.path.exists(self.hps.general.pre_trained):
                step_or_model_path = self.hps.general.pre_trained
            elif self.hps.general.resume_step == 'best':
                step_or_model_path = os.path.join(
                    self.result_subdir, get_best_model_name())
            elif self.hps.general.resume_step == 'latest':
                step_or_model_path = os.path.join(
                    self.result_subdir, get_latest_model_name(self.result_subdir))
            elif self.hps.general.resume_step != '':
                step_or_model_path = os.path.join(
                    self.result_subdir,
                    get_model_name(int(self.hps.general.resume_step)))

            if step_or_model_path is not None:
                state = load_snapshot(graph, step_or_model_path)

        return state

    def save(self,
             graph,
             epoch,
             model_name=None,
             is_best=False,
             state=None):
        save_snapshot(graph=graph,
                      epoch=epoch,
                      result_subdir=self.result_subdir,
                      model_name=model_name,
                      is_best=is_best,
                      state=state)

    def _init_subdir(self):
        result_subdir = None
        # resume subdir by run id
        if self.hps.general.warm_start and self.hps.general.resume_run_id is not None:
            result_subdir = locate_result_subdir(
                self.hps.general.result_dir, self.hps.general.resume_run_id)
        # create subdir
        if self.training and result_subdir is None:
            result_subdir = create_result_subdir(
                self.hps.general.result_dir,
                desc=self.hps.experiment,
                profile=self.hps,
                copy=True)
        return result_subdir
