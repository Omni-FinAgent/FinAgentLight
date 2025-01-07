import os
from argparse import Namespace

from mmengine import Config

from finagentlight.logger import logger
from finagentlight.utils.file_utils import assemble_project_path


def build_config(config_path: str, args: Namespace) -> Config:
    config = Config.fromfile(filename=config_path)

    if args.cfg_options is None:
        cfg_options = dict()
    else:
        cfg_options = dict()
        for cfg_option in args.cfg_options:
            key, value = cfg_option.split('=')
            cfg_options[key] = value
    for item in args.__dict__:
        if item not in ['config', 'cfg_options'] and args.__dict__[item] is not None:
            cfg_options[item] = args.__dict__[item]

    config.merge_from_dict(cfg_options)

    config.exp_path = assemble_project_path(os.path.join(config.workdir, config.tag))
    if config.if_remove is None:
        config.if_remove = bool(
            input(f"| Arguments PRESS 'y' to REMOVE: {config.exp_path}? ") == 'y'
        )
    if config.if_remove:
        import shutil

        shutil.rmtree(config.exp_path, ignore_errors=True)
        logger.info(f'| Arguments Remove work_dir: {config.exp_path}')
    else:
        logger.info(f'| Arguments Keep work_dir: {config.exp_path}')
    os.makedirs(config.exp_path, exist_ok=True)

    return config
