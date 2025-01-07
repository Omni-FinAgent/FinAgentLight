import sys
import warnings

warnings.filterwarnings('ignore')
import argparse
import os
import pathlib

from dotenv import load_dotenv
from mmengine import DictAction

load_dotenv(verbose=True)

root = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(root)

from finagentlight.config import build_config
from finagentlight.logger import logger
from finagentlight.registry import AGENT, DATASET, ENVIRONMENT, LLM
from finagentlight.utils.file_utils import assemble_project_path


def get_args_parser():
    parser = argparse.ArgumentParser(description='Crawler')
    parser.add_argument(
        '--config', default=os.path.join('configs', 'AAPL.py'), help='Config file path'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )

    parser.add_argument('--workdir', type=str, default='workdir')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--exp-path', type=str, default=None)
    parser.add_argument('--if_remove', action='store_true', default=False)

    return parser


def main(args):
    # 1. build config
    config = build_config(assemble_project_path(args.config), args)

    # 2. init dataset
    dataset = DATASET.build(config.dataset)

    # 3. init environment
    env_cfg = config.environment
    env_cfg['dataset'] = dataset
    environment = ENVIRONMENT.build(config.environment)

    # 4. llm
    llm = LLM.build(config.llm)

    # 5. init agent
    config.agent['llm'] = llm
    agent = AGENT.build(config.agent)
    agent.reset()

    state, info = environment.reset()
    logger.info(
        f'Timestamp: {info['timestamp']}, '
        f'Cash: {info['cash']}, '
        f'Position: {info['position']}, '
        f'Total Profit: {info['total_profit']}\n'
    )

    # 6. run
    while True:
        action = agent.step(state)
        if 'decision_making_decision' in action:
            state, reward, done, truncted, info = environment.step(
                action['decision_making_decision']
            )
            state.update(action)
            logger.info(
                f'Timestamp: {info['timestamp']}, '
                f'Cash: {info['cash']}, '
                f'Position: {info['position']}, '
                f'Total Profit: {info['total_profit']}.\n'
            )
        else:
            continue
        if done:
            break


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
