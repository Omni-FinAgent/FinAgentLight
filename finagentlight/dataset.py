import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch.utils.data

pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_rows', 100000)

from typing import Any, Dict, List, Tuple

from finagentlight.registry import DATASET, SCALER
from finagentlight.utils import json
from finagentlight.utils.file_utils import assemble_project_path
from finagentlight.utils.joblib_utils import load_joblib, save_joblib
from finagentlight.utils.timestamp import convert_timestamp_to_int

__all__ = ['Dataset']


@DATASET.register_module(force=True)
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *args,
        data_path: str,
        assets_path: str,
        fields_name: Dict[str, List[str]],
        if_norm: bool = True,
        if_norm_temporal: bool = False,
        if_use_temporal: bool = True,
        if_use_future: bool = True,
        scaler_cfg: Dict[str, Any],
        scaler_file: str,
        data_file: str,
        meta_info_file: str,
        history_timestamps: int = 64,
        future_timestamps: int = 32,
        start_timestamp: str,
        end_timestamp: str,
        timestamp_format: str = '%Y-%m-%d',
        exp_path: str,
        **kwargs,
    ):
        super(Dataset, self).__init__()

        self.data_path = assemble_project_path(data_path)

        self.assets_path = assemble_project_path(assets_path)

        self.fields_name = fields_name

        self.features_name = self.fields_name['features']
        self.prices_name = self.fields_name['prices']
        self.temporals_name = self.fields_name['temporals']
        self.labels_name = self.fields_name['labels']

        self.if_norm = if_norm
        self.if_norm_temporal = if_norm_temporal
        self.if_use_temporal = if_use_temporal
        self.if_use_future = if_use_future

        exp_path = assemble_project_path(exp_path)
        self.scaler_file = os.path.join(exp_path, scaler_file)
        self.data_file = os.path.join(exp_path, data_file)
        self.meta_info_file = os.path.join(exp_path, meta_info_file)

        self.history_timestamps = history_timestamps
        self.future_timestamps = future_timestamps

        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

        self.timestamp_format = timestamp_format

        self.scaler_cfg = scaler_cfg
        if os.path.exists(self.scaler_file):
            self.scalers = load_joblib(self.scaler_file)
        else:
            self.scalers = None

        self.assets = self._init_assets_info()
        self.features_df_dict, self.news_df_dict = self._load_assets_df(self.assets)

        if os.path.exists(self.data_file):
            self.data = load_joblib(self.data_file)
            self.meta_info = load_joblib(self.meta_info_file)
        else:
            self.data = self._init_data(
                self.assets, self.features_df_dict, self.news_df_dict, self.scalers
            )
            self.meta_info = self._init_meta_info(
                self.assets, self.features_df_dict, self.news_df_dict
            )

            save_joblib(self.scalers, self.scaler_file)
            save_joblib(self.data, self.data_file)
            save_joblib(self.meta_info, self.meta_info_file)

    def _init_assets_info(self):
        assets = json.load(self.assets_path)
        assets = OrderedDict(sorted(assets.items(), key=lambda x: x[0]))
        for key, value in assets.items():
            value.update({'type': 'stock'})
            assets[key] = value
        return assets

    def _load_assets_df(
        self, assets: List[str]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        start_timestamp = (
            pd.to_datetime(self.start_timestamp, format=self.timestamp_format)
            if self.start_timestamp
            else None
        )
        end_timestamp = (
            pd.to_datetime(self.end_timestamp, format=self.timestamp_format)
            if self.end_timestamp
            else None
        )

        features_df_dict = {}
        news_df_dict = {}
        for asset in assets:
            feature_path = os.path.join(
                self.data_path, 'features', '{}.parquet'.format(asset)
            )
            assert os.path.exists(
                feature_path
            ), f'feature file {feature_path} not exists'

            news_path = os.path.join(self.data_path, 'news', '{}.parquet'.format(asset))
            assert os.path.exists(news_path), f'news file {news_path} not exists'

            feature_df = pd.read_parquet(feature_path)
            feature_df.set_index('timestamp', inplace=True, drop=True)
            feature_df.index = pd.to_datetime(feature_df.index).normalize()
            news_df = pd.read_parquet(news_path)
            news_df.set_index('timestamp', inplace=True, drop=True)
            news_df.index = pd.to_datetime(news_df.index).normalize()

            if start_timestamp and end_timestamp:
                feature_df = feature_df.loc[start_timestamp:end_timestamp]
                news_df = news_df.loc[start_timestamp:end_timestamp]
            elif start_timestamp:
                feature_df = feature_df.loc[start_timestamp:]
                news_df = news_df.loc[start_timestamp:]
            elif end_timestamp:
                feature_df = feature_df.loc[:end_timestamp]
                news_df = news_df.loc[:end_timestamp]

            features_df_dict[asset] = feature_df
            news_df_dict[asset] = news_df

        return features_df_dict, news_df_dict

    def _init_data(
        self,
        assets: Dict[str, Any],
        features_df_dict: Dict[str, pd.DataFrame],
        news_df_dict: Dict[str, pd.DataFrame],
        scalers: Dict[str, Any],
    ):
        datas = {}
        self.scalers = {}

        for asset in assets:
            feature_df = features_df_dict[asset]
            news_df = news_df_dict[asset]
            timestamp = feature_df.index

            price_indices = [
                self.features_name.index(price_name) for price_name in self.prices_name
            ]
            original_price = feature_df[self.prices_name]

            if self.if_norm:
                if self.scalers is None or asset not in self.scalers:
                    scaler = SCALER.build(self.scaler_cfg)

                    if self.if_norm_temporal:
                        feature_df[self.features_name + self.temporals_name] = (
                            scaler.fit_transform(
                                feature_df[self.features_name + self.temporals_name]
                            )
                        )
                    else:
                        feature_df[self.features_name] = scaler.fit_transform(
                            feature_df[self.features_name]
                        )
                else:
                    scaler = self.scalers[asset]
                    if self.if_norm_temporal:
                        feature_df[self.features_name + self.temporals_name] = (
                            scaler.transform(
                                feature_df[self.features_name + self.temporals_name]
                            )
                        )
                    else:
                        feature_df[self.features_name] = scaler.transform(
                            feature_df[self.features_name]
                        )

                if len(scaler.mean.shape) == 1 and len(scaler.std.shape) == 1:
                    price_mean = scaler.mean[None, 1].repeat(
                        feature_df.shape[0], axis=0
                    )[..., price_indices]
                    price_std = scaler.std[None, 1].repeat(feature_df.shape[0], axis=0)[
                        ..., price_indices
                    ]
                else:
                    price_mean = scaler.mean[..., price_indices]
                    price_std = scaler.std[..., price_indices]

            else:
                scaler = None
                price_mean = np.zeros((feature_df.shape[0], len(self.prices_name)))
                price_std = np.ones((feature_df.shape[0], len(self.prices_name)))

            self.scalers[asset] = scaler

            if self.if_use_temporal:
                feature = feature_df[self.features_name + self.temporals_name]
            else:
                feature = feature_df[self.features_name]

            price = feature_df[self.prices_name]
            label = feature_df[self.labels_name]

            news_df['id'] = pd.Series(range(len(news_df)))
            news = news_df

            data = dict(
                timestamp=timestamp,
                feature=feature,
                label=label,
                price=price,
                news=news,
                original_price=original_price,
                price_mean=price_mean,
                price_std=price_std,
            )

            datas[asset] = data

        return datas

    def _get_index(
        self,
        df: pd.DataFrame,
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
    ) -> Tuple[int, int]:
        filtered_df = df.loc[start_timestamp:end_timestamp]

        if filtered_df.empty:
            return -1, -1

        start_indices = df.index.get_loc(filtered_df.index[0])
        end_indices = df.index.get_loc(filtered_df.index[-1])

        start_index = (
            start_indices.start if isinstance(start_indices, slice) else start_indices
        )
        end_index = (
            end_indices.stop - 1 if isinstance(end_indices, slice) else end_indices
        )

        return start_index, end_index

    def _init_meta_info(
        self,
        assets: Dict[str, Any],
        features_df_dict: Dict[str, pd.DataFrame],
        news_df_dict: Dict[str, pd.DataFrame],
    ):
        data_infos = {}

        assets_name = list(assets.keys())
        first_asset = features_df_dict[assets_name[0]]
        future_timestamps = self.future_timestamps if self.if_use_future else 0

        for asset in assets:
            data_info: Dict[int, Any] = {}

            count = 0
            feature_df = features_df_dict[asset]
            news_df = news_df_dict[asset]

            for i in range(
                self.history_timestamps, len(first_asset) - future_timestamps
            ):
                data_info[count] = {}

                start_timestamp = feature_df.index[i - self.history_timestamps]
                end_timestamp = feature_df.index[i - 1]

                feature_start_index, feature_end_index = self._get_index(
                    feature_df, start_timestamp, end_timestamp
                )
                news_start_index, news_end_index = self._get_index(
                    news_df, start_timestamp, end_timestamp
                )

                history_info = {
                    'start_timestamp': start_timestamp,
                    'end_timestamp': end_timestamp,
                    'feature_start_index': feature_start_index,
                    'feature_end_index': feature_end_index,
                    'news_start_index': news_start_index,
                    'news_end_index': news_end_index,
                }

                data_info[count].update({'history_info': history_info})

                if self.if_use_future:
                    start_timestamp = feature_df.index[i]
                    end_timestamp = feature_df.index[i + self.future_timestamps - 1]

                    feature_start_index, feature_end_index = self._get_index(
                        feature_df, start_timestamp, end_timestamp
                    )
                    news_start_index, news_end_index = self._get_index(
                        news_df, start_timestamp, end_timestamp
                    )

                    future_info = {
                        'start_timestamp': start_timestamp,
                        'end_timestamp': end_timestamp,
                        'feature_start_index': feature_start_index,
                        'feature_end_index': feature_end_index,
                        'news_start_index': news_start_index,
                        'news_end_index': news_end_index,
                    }

                    data_info[count].update({'future_info': future_info})

                count += 1

            data_infos[asset] = data_info

        return data_infos

    def __str__(self):
        str = f'{'-' * 50} Dataset {'-' * 50}\n'
        for asset in self.assets:
            str += f'asset: {asset}\n'
            str += f'timestamp start: {self.data[asset]['timestamp'][0]}, end: {self.data[asset]['timestamp'][-1]}\n'
            str += f'feature shape: {self.data[asset]['feature'].shape}\n'
            str += f'label shape: {self.data[asset]['label'].shape}\n'
            str += f'news shape: {self.data[asset]['news'].shape}\n'
            str += f'price shape: {self.data[asset]['price'].shape}\n'
            str += f'price mean shape: {self.data[asset]['price_mean'].shape}\n'
            str += f'price std shape: {self.data[asset]['price_std'].shape}\n'
            if self.scalers[asset]:
                str += f'scaler: mean: {self.scalers[asset].mean.shape}, std: {self.scalers[asset].std.shape}\n'
            else:
                str += 'scaler: None\n'
            str += '\n'
        str += f'{'-' * 50} Dataset {'-' * 50}\n'
        return str

    def __len__(self):
        return len(self.meta_info)

    def __getitem__(self, idx):
        items = {}

        for asset in self.assets:
            asset_meta_info = self.meta_info[asset][idx]

            history_info = asset_meta_info['history_info']
            history_data = {
                'start_timestamp': convert_timestamp_to_int(
                    history_info['start_timestamp']
                ),
                'end_timestamp': convert_timestamp_to_int(
                    history_info['end_timestamp']
                ),
                'feature_start_index': history_info['feature_start_index'],
                'feature_end_index': history_info['feature_end_index'],
                'news_start_index': history_info['news_start_index'],
                'news_end_index': history_info['news_end_index'],
                'feature': self.data[asset]['feature']
                .loc[history_info['start_timestamp'] : history_info['end_timestamp']]
                .values.astype('float32'),
                'label': self.data[asset]['label']
                .loc[history_info['start_timestamp'] : history_info['end_timestamp']]
                .values.astype('float32'),
                'price': self.data[asset]['price']
                .loc[history_info['start_timestamp'] : history_info['end_timestamp']]
                .values.astype('float32'),
                'news': self.data[asset]['news']
                .loc[history_info['start_timestamp'] : history_info['end_timestamp']]
                .to_dict(orient='records'),
                'original_price': self.data[asset]['original_price']
                .loc[history_info['start_timestamp'] : history_info['end_timestamp']]
                .values.astype('float32'),
                'price_mean': self.data[asset]['price_mean'][
                    history_info['feature_start_index'] : history_info[
                        'feature_end_index'
                    ]
                    + 1,
                    ...,
                ].astype('float32'),
                'price_std': self.data[asset]['price_std'][
                    history_info['feature_start_index'] : history_info[
                        'feature_end_index'
                    ]
                    + 1,
                    ...,
                ].astype('float32'),
                'timestamp': self.data[asset]['feature']
                .loc[history_info['start_timestamp'] : history_info['end_timestamp']]
                .reset_index(drop=False)['timestamp']
                .apply(lambda x: convert_timestamp_to_int(x))
                .values.astype('float32'),
                'description': ['sample text'] * len(self.assets),
            }

            item = {
                'asset': asset,
                'history': history_data,
            }

            if self.if_use_future:
                future_info = asset_meta_info['future_info']
                future_data = {
                    'start_timestamp': convert_timestamp_to_int(
                        future_info['start_timestamp']
                    ),
                    'end_timestamp': convert_timestamp_to_int(
                        future_info['end_timestamp']
                    ),
                    'feature_start_index': future_info['feature_start_index'],
                    'feature_end_index': future_info['feature_end_index'],
                    'news_start_index': future_info['news_start_index'],
                    'news_end_index': future_info['news_end_index'],
                    'feature': self.data[asset]['feature']
                    .loc[future_info['start_timestamp'] : future_info['end_timestamp']]
                    .values.astype('float32'),
                    'label': self.data[asset]['label']
                    .loc[future_info['start_timestamp'] : future_info['end_timestamp']]
                    .values.astype('float32'),
                    'price': self.data[asset]['price']
                    .loc[future_info['start_timestamp'] : future_info['end_timestamp']]
                    .values.astype('float32'),
                    'news': self.data[asset]['news']
                    .loc[future_info['start_timestamp'] : future_info['end_timestamp']]
                    .to_dict(orient='records'),
                    'original_price': self.data[asset]['original_price']
                    .loc[future_info['start_timestamp'] : future_info['end_timestamp']]
                    .values.astype('float32'),
                    'price_mean': self.data[asset]['price_mean'][
                        future_info['feature_start_index'] : future_info[
                            'feature_end_index'
                        ]
                        + 1,
                        ...,
                    ].astype('float32'),
                    'price_std': self.data[asset]['price_std'][
                        future_info['feature_start_index'] : future_info[
                            'feature_end_index'
                        ]
                        + 1,
                        ...,
                    ].astype('float32'),
                    'timestamp': self.data[asset]['feature']
                    .loc[future_info['start_timestamp'] : future_info['end_timestamp']]
                    .reset_index(drop=False)['timestamp']
                    .apply(lambda x: convert_timestamp_to_int(x))
                    .values.astype('float32'),
                    'description': ['sample text'] * len(self.assets),
                }

                item.update({'future': future_data})

            items[asset] = item

        return items


if __name__ == '__main__':
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
            window_size=64,
        ),
        scaler_file='scalers.joblib',
        data_file='scaled_data.joblib',
        meta_info_file='meta_info.joblib',
        history_timestamps=64,
        future_timestamps=32,
        start_timestamp='2020-01-02',
        end_timestamp='2023-12-29',
        timestamp_format='%Y-%m-%d',
        exp_path=assemble_project_path(os.path.join('workdir', 'tmp')),
    )

    dataset = DATASET.build(dataset)
    print(dataset)
    print(len(dataset))
