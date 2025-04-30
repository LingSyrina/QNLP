import pandas as pd
import numpy as np
import argparse
import os

def main(input_file):
    # Set random seed for reproducibility
    SEED = 12
    np.random.seed(SEED)

    # Output files
    train_file = 'mc_pair_train_data.csv'
    dev_file = 'mc_pair_dev_data.csv'
    test_file = 'mc_pair_test_data.csv'

    # Load the dataset
    data = pd.read_csv(input_file, header=None)

    # Shuffle the data
    # data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Randomly select 20 rows
    random_rows = data.sample(n=100, random_state=SEED).reset_index(drop=True)

    # Split proportions
    train_frac = 0.7
    dev_frac = 0.15
    test_frac = 0.15

    # Calculate split indices
    train_end = int(train_frac * len(random_rows))
    dev_end = train_end + int(dev_frac * len(random_rows))

    # Split into train, dev, test
    train_data = random_rows.iloc[:train_end]
    dev_data = random_rows.iloc[train_end:dev_end]
    test_data = random_rows.iloc[dev_end:]

    # n_total = len(data)
    # n_train = int(train_frac * n_total)
    # n_dev = int(dev_frac * n_total)
    #
    # # Split the data
    # train_data = data.iloc[:n_train]
    # dev_data = data.iloc[n_train:n_train+n_dev]
    # test_data = data.iloc[n_train+n_dev:]

    # Save to CSVs
    train_data.to_csv(train_file, index=False, header=False)
    dev_data.to_csv(dev_file, index=False, header=False)
    test_data.to_csv(test_file, index=False, header=False)

    print(f"Data split complete:\nTrain: {len(train_data)} samples\nDev: {len(dev_data)} samples\nTest: {len(test_data)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, dev, and test sets.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    args = parser.parse_args()

    main(args.input_file)
