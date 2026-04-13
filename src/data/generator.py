import numpy as np
import pandas as pd
import os
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()['parameters']

def generate_normal(n):
    data = {}
    for param, rules in config.items():
        mean = (rules['min_normal'] + rules['max_normal']) / 2
        std = (rules['max_normal'] - rules['min_normal']) / 6
        values = np.random.normal(mean, std, n)
        values = np.clip(values, rules['min_normal'], rules['max_normal'])
        data[param] = values

    logger.info(f'Generated {n} normal patient records')
    return pd.DataFrame(data)

def generate_high_risk(n):
    data = {}
    for param, rules in config.items():
        mask = np.random.rand(n) > 0.5
        high_val = np.random.uniform(
            rules['critical_high'], rules['absolute_max'], n
        )
        low_val = np.random.uniform(
            rules['absolute_min'], rules['critical_low'], n
        )
        data[param] = np.where(mask, high_val, low_val)

    logger.info(f'Generated {n} high-risk patient records')
    return pd.DataFrame(data)

def generate_broken(n):
    data = {}
    for param, rules in config.items():
        mask = np.random.rand(n) > 0.5
        too_high = np.random.uniform(
            rules['absolute_max'] + 0.1,
            rules['absolute_max'] * 3,
            n
        )
        broken_floor = min(rules['absolute_min'] - 1.0, -1.0)
        too_low = np.random.uniform(
            broken_floor - 10.0,
            rules['absolute_min'] - 0.1,
            n
        )
        data[param] = np.where(mask, too_high, too_low)

    logger.info(f'Generated {n} broken/corrupt patient records')
    return pd.DataFrame(data)

def generate_dataset(n=5000):
    n_normal = int(n * 0.80)   # 4000 records
    n_risk = int(n * 0.15)      # 750 records
    n_broken = int(n * 0.05)    # 250 records

    df_normal = generate_normal(n_normal)
    df_normal['label'] = 0

    df_risk = generate_high_risk(n_risk)
    df_risk['label'] = 1

    df_broken = generate_broken(n_broken)
    df_broken['label'] = -1

    df = pd.concat([df_normal, df_risk, df_broken], ignore_index=True)
    logger.info(f'Final dataset: {len(df)} total rows | Normal: {n_normal} | Risk: {n_risk} | Broken: {n_broken}')
    return df

if __name__ == '__main__':
    df = generate_dataset(5000)

    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/synthetic_data.csv', index=False)
    logger.info('Dataset saved to data/raw/synthetic_data.csv')

    print('\nDataset Summary:')
    print(f'Total rows: {len(df)}')
    print(f'Label distribution:\n{df["label"].value_counts()}')
    print(f'\nFirst 3 rows:\n{df.head(3)}')