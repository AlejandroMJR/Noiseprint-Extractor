import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(21)

vision_path = "/nas/public/dataset/Vision"
dresden_path = "/nas/public/dataset/dresden/natural/jpeg"
ieee_path = "/nas/public/dataset/ieee-spcup-2018"
save_path = "/nas/home/ajaramillo/projects/Try"


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--test', help='test percentage over the total amount of data', type=float, default=0.2)
    parser.add_argument('--val', help='val percentage over the amount of train data', type=float, default=0.2)

    args = parser.parse_args()
    #test_frac = args.test
    val_frac = args.val
    train_frac = 1-val_frac

    # read dataframe
    df_vision = pd.read_csv(os.path.join(vision_path, "VisionNaturalDataset.csv"), sep="|")

    #correct path to include full path
    for idx, row in df_vision.iterrows():
        df_vision["probe"].values[idx] = os.path.join(vision_path, df_vision["probe"].values[idx])

    df_dresden = pd.read_csv(os.path.join(dresden_path, "DresdenJpegNaturalDataset.csv"), sep="|")
    for idx, row in df_dresden.iterrows():
        df_dresden["probe"].values[idx] = os.path.join(dresden_path, df_dresden["probe"].values[idx])

    df_ieee = pd.read_csv(os.path.join(save_path, "ieeeDataset.csv"), sep=",")

    df = pd.concat([df_vision, df_dresden, df_ieee])

    classes = df['model'].unique()
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    #test_data = pd.DataFrame()
    for c in classes:
        class_data = df.loc[df['model'] == c]

        # Get the number of samples in the class
        num_samples = len(class_data)

        # Randomly shuffle the data
        #class_data = class_data.sample(frac=1, random_state=42)

        # Split the data into train, val, and test sets
        #train_data = train_data.append(class_data[:int(num_samples * train_frac)])
        train_data = pd.concat([train_data, class_data[:int(num_samples * train_frac)]], ignore_index=True)
        val_data = pd.concat([val_data, class_data[int(num_samples * train_frac):int(num_samples * (train_frac + val_frac))]], ignore_index=True)
        #test_data = pd.concat([test_data, class_data[int(num_samples * (train_frac + val_frac)):]])

    train_data.index.name = "id"
    val_data.index.name = "id"
    # Save pickles
    train_data.to_csv(os.path.join(save_path, 'train.csv'))
    val_data.to_csv(os.path.join(save_path, 'val.csv'))
    #test_data.to_csv(os.path.join(save_path, 'test_v_d.csv'))

    return 0


if __name__ == '__main__':
    main()

