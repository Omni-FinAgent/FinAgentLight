import os
import warnings

warnings.filterwarnings('ignore')

from typing import Any

import gym
import numpy as np
import pandas as pd

from finagentlight.registry import DATASET, ENVIRONMENT
from finagentlight.utils.file_utils import assemble_project_path

__all__ = ['Environment']


@ENVIRONMENT.register_module(force=True)
class Environment(gym.Env):
    def __init__(
        self,
        *args,
        dataset: Any = None,
        select_asset: str = '',
        initial_amount: float = 1e3,
        transaction_cost_pct: float = 1e-3,
        timestamp_format: str = '%Y-%m-%d',
        history_timestamps: int = 32,
        step_timestamps: int = 1,
        future_timestamps: int = 32,
        start_timestamp='2008-04-01',
        end_timestamp='2021-04-01',
        gamma: float = 0.99,
        **kwargs,
    ):
        super(Environment, self).__init__()

        self.dataset = dataset
        self.assets = self.dataset.assets
        self.select_asset = select_asset
        assert (
            self.select_asset in self.assets and self.select_asset is not None
        ), f'select_asset {self.select_asset} not in assets {self.assets}'

        asset_info = self.dataset.assets[self.select_asset]

        self.asset_info = dict(
            asset_symbol=asset_info['symbol'],
            asset_name=asset_info['companyName'],
            asset_type=asset_info['type'],
            asset_exchange=asset_info['exchange'],
            asset_sector=asset_info['sector'],
            asset_industry=asset_info['industry'],
            asset_description=asset_info['description'],
        )

        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct

        self.data = self.dataset.data[self.select_asset]
        self.meta_info = self.dataset.meta_info[self.select_asset]

        self.prices_name = self.dataset.prices_name
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.history_timestamps = history_timestamps
        self.step_timestamps = step_timestamps
        self.future_timestamps = future_timestamps
        self.timestamp_format = timestamp_format
        self.gamma = gamma

        res_info = self._init_features()
        self.timestamp_info = res_info['timestamp_info']
        self.features_df = res_info['features_df']
        self.prices_df = res_info['prices_df']
        self.news_df = res_info['news_df']

        self.action_labels = ['SELL', 'HOLD', 'BUY']  # 0, 1, 2
        self.action_dim = len(self.action_labels)

    def _init_features(self):
        timestamp_info = {}
        for key, value in self.meta_info.items():
            history_start_timestamp = value['history_info']['start_timestamp']
            history_end_timestamp = value['history_info']['end_timestamp']
            future_start_timestamp = value['future_info']['start_timestamp']
            future_end_timestamp = value['future_info']['end_timestamp']

            if history_start_timestamp >= pd.to_datetime(
                self.start_timestamp
            ) and future_end_timestamp < pd.to_datetime(self.end_timestamp):
                timestamp_info[key] = {
                    'history_start_timestamp': history_start_timestamp,
                    'history_end_timestamp': history_end_timestamp,
                    'future_start_timestamp': future_start_timestamp,
                    'future_end_timestamp': future_end_timestamp,
                }

        self.timestamp_min_index = min(timestamp_info.keys())
        self.timestamp_max_index = max(timestamp_info.keys())
        self.timestamp_min = timestamp_info[self.timestamp_min_index][
            'history_end_timestamp'
        ]
        self.timestamp_max = timestamp_info[self.timestamp_max_index][
            'history_end_timestamp'
        ]

        self.num_timestamps = self.timestamp_max_index - self.timestamp_min_index + 1
        assert (
            self.num_timestamps == len(timestamp_info)
        ), f'num_timestamps {self.num_timestamps} != len(data_info) {len(timestamp_info)}'

        features_df = self.data['feature']
        prices_df = self.data['original_price']
        news_df = self.data['news']

        res_info = dict(
            timestamp_info=timestamp_info,
            features_df=features_df,
            prices_df=prices_df,
            news_df=news_df,
        )

        return res_info

    def _init_timestamp_index(self):
        timestamp_index = self.timestamp_min_index
        return timestamp_index

    def get_timestamp(self, timestamp_index: int):
        return self.timestamp_info[timestamp_index]['history_end_timestamp']

    def get_value(self, cash: float, postition: int, price: float):
        value = cash + postition * price
        return value

    def get_price(self, timestamp_index: int):
        timestamp_info = self.timestamp_info[timestamp_index]
        end_timestamp = timestamp_info['history_end_timestamp']

        next_timestamp_info = self.timestamp_info[
            timestamp_index + 1
            if timestamp_index + 1 <= self.timestamp_max_index
            else timestamp_index
        ]
        next_end_timestamp = next_timestamp_info['history_end_timestamp']

        prices = self.prices_df.loc[end_timestamp].values
        next_prices = self.prices_df.loc[next_end_timestamp].values

        _, _, _, _, adj = prices[0], prices[1], prices[2], prices[3], prices[4]
        price = adj
        _, _, _, _, next_adj = (
            next_prices[0],
            next_prices[1],
            next_prices[2],
            next_prices[3],
            next_prices[4],
        )
        next_price = next_adj

        return price, next_price

    def get_state(self, timestamp_index: int):
        timestamp_info = self.timestamp_info[timestamp_index]

        history_start_timestamp = timestamp_info['history_start_timestamp']
        history_end_timestamp = timestamp_info['history_end_timestamp']
        future_start_timestamp = timestamp_info['future_start_timestamp']
        future_end_timestamp = timestamp_info['future_end_timestamp']

        history_price = self.prices_df.loc[
            history_start_timestamp:history_end_timestamp
        ]
        history_news = self.news_df.loc[history_start_timestamp:history_end_timestamp]
        future_price = self.prices_df.loc[future_start_timestamp:future_end_timestamp]
        future_news = self.news_df.loc[future_start_timestamp:future_end_timestamp]

        state = dict(
            timestamp=history_end_timestamp,
            history_price=history_price,
            history_news=history_news,
            future_price=future_price,
            future_news=future_news,
        )
        state.update(self.asset_info)

        return state

    def eval_buy_position(self, cash: float, price: float):
        # evaluate buy position
        # price * position + price * position * transaction_cost_pct <= cash
        # position <= cash / price / (1 + transaction_cost_pct)
        return int(np.floor(cash / price / (1 + self.transaction_cost_pct)))

    def eval_sell_position(self, position: int):
        # evaluate sell position
        return int(position)

    def buy(self, cash: float, position: int, price: float, amount: int):
        # evaluate buy position
        eval_buy_postion = self.eval_buy_position(price=price, cash=cash)

        # predict buy position
        buy_position = int(np.floor((1.0 * np.abs(amount)) * eval_buy_postion))

        cash = cash - (buy_position * price * (1 + self.transaction_cost_pct))
        position = position + buy_position
        value = self.get_value(cash=cash, postition=position, price=price)

        if buy_position == 0:
            action_label = 'HOLD'
            action = self.action_labels.index('HOLD')
        else:
            action_label = 'BUY'
            action = self.action_labels.index('BUY')

        res_info = dict(
            cash=cash,
            position=position,
            value=value,
            action=action,
            action_label=action_label,
        )
        return res_info

    def sell(self, cash: float, position: int, price: float, amount: int):
        # evaluate sell position
        eval_sell_postion = self.eval_sell_position(position=position)

        # predict sell position
        sell_position = int(np.floor((1.0 * np.abs(amount)) * eval_sell_postion))

        cash = cash + (sell_position * price * (1 - self.transaction_cost_pct))
        position = position - sell_position
        value = self.get_value(cash=cash, postition=position, price=price)

        if sell_position == 0:
            action_label = 'HOLD'
            action = self.action_labels.index('HOLD')
        else:
            action_label = 'SELL'
            action = self.action_labels.index('SELL')

        res_info = dict(
            cash=cash,
            position=position,
            value=value,
            action=action,
            action_label=action_label,
        )

        return res_info

    def hold(self, cash: float, position: int, price: float, amount: int):
        value = self.get_value(cash=cash, postition=position, price=price)

        action_label = 'HOLD'
        action = self.action_labels.index('HOLD')

        res_info = dict(
            cash=cash,
            position=position,
            value=value,
            action=action,
            action_label=action_label,
        )

        return res_info

    def reset(self, **kwargs):
        self.timestamp_index = self._init_timestamp_index()
        self.timestamp = self.get_timestamp(timestamp_index=self.timestamp_index)

        self.state = self.get_state(timestamp_index=self.timestamp_index)

        self.price, self.next_price = self.get_price(
            timestamp_index=self.timestamp_index
        )

        self.ret = 0.0
        self.cash = self.initial_amount
        self.position = 0
        self.discount = 1.0
        self.value = self.initial_amount
        self.total_return = 0.0
        self.total_profit = 0.0
        self.action = 1
        self.action_label = 'HOLD'
        self.done = False

        info = dict(
            timestamp=self.timestamp.strftime(self.timestamp_format),
            ret=self.ret,
            price=self.price,
            next_price=self.next_price,
            cash=self.cash,
            position=self.position,
            discount=self.discount,
            value=self.value,
            total_profit=self.total_profit,
            total_return=self.total_return,
            action=self.action,
            action_label=self.action_label,
            done=self.done,
        )

        return self.state, info

    def step(self, action: int | str):
        if isinstance(action, str):
            action = self.action_labels.index(action)

        action = action - 1  # modify the action to -1, 0, 1

        if action > 0:
            res_info = self.buy(
                cash=self.cash, position=self.position, price=self.price, amount=action
            )
        elif action < 0:
            res_info = self.sell(
                cash=self.cash, position=self.position, price=self.price, amount=action
            )
        else:
            res_info = self.hold(
                cash=self.cash, position=self.position, price=self.price, amount=action
            )

        self.cash = res_info['cash']
        self.position = res_info['position']
        self.value = res_info['value']
        self.action = res_info['action']
        self.action_label = res_info['action_label']

        self.timestamp_index = self.timestamp_index + 1
        self.timestamp = self.get_timestamp(timestamp_index=self.timestamp_index)

        self.state = self.get_state(timestamp_index=self.timestamp_index)

        self.price, self.next_price = self.get_price(
            timestamp_index=self.timestamp_index
        )

        pre_value = self.get_value(
            cash=self.cash, postition=self.position, price=self.price
        )

        next_value = self.get_value(
            cash=self.cash, postition=self.position, price=self.next_price
        )

        if self.timestamp_index < self.timestamp_max_index:
            self.done = False
            self.truncted = False
        else:
            self.done = True
            self.truncted = True

        self.next_state = self.state

        reward = (next_value - pre_value) / pre_value

        self.state = self.next_state

        self.ret = reward
        self.discount *= 0.99
        self.total_return += self.discount * reward
        self.total_profit = (
            (self.value - self.initial_amount) / self.initial_amount * 100
        )

        info = dict(
            timestamp=self.timestamp.strftime(self.timestamp_format),
            ret=self.ret,
            price=self.price,
            next_price=self.next_price,
            cash=self.cash,
            position=self.position,
            discount=self.discount,
            value=self.value,
            total_profit=self.total_profit,
            total_return=self.total_return,
            action=self.action,
            action_label=self.action_label,
            done=self.done,
        )

        return self.next_state, reward, self.done, self.truncted, info


if __name__ == '__main__':
    select_asset = 'AAPL'

    dataset = dict(
        type='Dataset',
        data_path='datasets/exp_stocks',
        assets_path='configs/_asset_list_/exp_stocks.json',
        fields_name={
            'features': [
                'open',
                'high',
                'low',
                'close',
                'adj_close',
                'kmid',
                'kmid2',
                'klen',
                'kup',
                'kup2',
                'klow',
                'klow2',
                'ksft',
                'ksft2',
                'roc_5',
                'roc_10',
                'roc_20',
                'roc_30',
                'roc_60',
                'ma_5',
                'ma_10',
                'ma_20',
                'ma_30',
                'ma_60',
                'std_5',
                'std_10',
                'std_20',
                'std_30',
                'std_60',
                'beta_5',
                'beta_10',
                'beta_20',
                'beta_30',
                'beta_60',
                'max_5',
                'max_10',
                'max_20',
                'max_30',
                'max_60',
                'min_5',
                'min_10',
                'min_20',
                'min_30',
                'min_60',
                'qtlu_5',
                'qtlu_10',
                'qtlu_20',
                'qtlu_30',
                'qtlu_60',
                'qtld_5',
                'qtld_10',
                'qtld_20',
                'qtld_30',
                'qtld_60',
                'rank_5',
                'rank_10',
                'rank_20',
                'rank_30',
                'rank_60',
                'imax_5',
                'imax_10',
                'imax_20',
                'imax_30',
                'imax_60',
                'imin_5',
                'imin_10',
                'imin_20',
                'imin_30',
                'imin_60',
                'imxd_5',
                'imxd_10',
                'imxd_20',
                'imxd_30',
                'imxd_60',
                'rsv_5',
                'rsv_10',
                'rsv_20',
                'rsv_30',
                'rsv_60',
                'cntp_5',
                'cntp_10',
                'cntp_20',
                'cntp_30',
                'cntp_60',
                'cntn_5',
                'cntn_10',
                'cntn_20',
                'cntn_30',
                'cntn_60',
                'cntd_5',
                'cntd_10',
                'cntd_20',
                'cntd_30',
                'cntd_60',
                'corr_5',
                'corr_10',
                'corr_20',
                'corr_30',
                'corr_60',
                'cord_5',
                'cord_10',
                'cord_20',
                'cord_30',
                'cord_60',
                'sump_5',
                'sump_10',
                'sump_20',
                'sump_30',
                'sump_60',
                'sumn_5',
                'sumn_10',
                'sumn_20',
                'sumn_30',
                'sumn_60',
                'sumd_5',
                'sumd_10',
                'sumd_20',
                'sumd_30',
                'sumd_60',
                'vma_5',
                'vma_10',
                'vma_20',
                'vma_30',
                'vma_60',
                'vstd_5',
                'vstd_10',
                'vstd_20',
                'vstd_30',
                'vstd_60',
                'wvma_5',
                'wvma_10',
                'wvma_20',
                'wvma_30',
                'wvma_60',
                'vsump_5',
                'vsump_10',
                'vsump_20',
                'vsump_30',
                'vsump_60',
                'vsumn_5',
                'vsumn_10',
                'vsumn_20',
                'vsumn_30',
                'vsumn_60',
                'vsumd_5',
                'vsumd_10',
                'vsumd_20',
                'vsumd_30',
                'vsumd_60',
            ],
            'prices': [
                'open',
                'high',
                'low',
                'close',
                'adj_close',
            ],
            'temporals': [
                'day',
                'weekday',
                'month',
            ],
            'labels': ['ret1', 'mov1'],
        },
        if_norm=True,
        if_norm_temporal=False,
        scaler_cfg=dict(
            type='WindowedScaler',
            window_size=14,
        ),
        scaler_file='scalers.joblib',
        data_file='scaled_data.joblib',
        meta_info_file='meta_info.joblib',
        history_timestamps=14,
        future_timestamps=14,
        start_timestamp='2020-01-02',
        end_timestamp='2023-12-29',
        timestamp_format='%Y-%m-%d',
        exp_path=assemble_project_path(os.path.join('workdir', 'tmp')),
    )

    env_cfg: dict[str, Any] = dict(
        type='Environment',
        dataset=None,
        select_asset=select_asset,
        initial_amount=float(1e5),
        transaction_cost_pct=float(1e-4),
        timestamp_format='%Y-%m-%d',
        start_timestamp='2020-04-02',
        end_timestamp='2023-12-29',
        history_timestamps=14,
        future_timestamps=14,
        step_timestamps=1,
    )

    dataset = DATASET.build(dataset)

    env_cfg.update(
        dict(
            dataset=dataset,
        )
    )

    environment = ENVIRONMENT.build(env_cfg)

    state, info = environment.reset()

    for step in range(900):
        action = np.random.choice([0, 1, 2])
        state, reward, done, truncted, info = environment.step(action)
        print(
            f'step: {step}, action: {action}, reward: {reward}, done: {done}, truncted: {truncted}, info: {info}'
        )
