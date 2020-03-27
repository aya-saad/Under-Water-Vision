import os
import sys

import pandas as pd
import split_folders


def dataset_info_to_csv(dir):
    classes = [name for name in os.listdir(dir) if name != ".DS_Store"]
    print(classes)
    data = []

    for label in classes:
        n = len([name for name in os.listdir("{}/{}".format(dir, label)) if
                 os.path.isfile(os.path.join(dir, label, name))])
        data.append([label, n])

    df = pd.DataFrame(data, columns=['Label', 'n_img'])
    df = df.append(df.sum(numeric_only=True), ignore_index=True)
    print(df)

    df.to_csv("{}_info.csv".format(dir.split("/")[0]))

def split_dataset_to_train_test_val(dir, out_dir):
    for f in os.listdir(dir):
        print(f)
    split_folders.ratio(dir, output=out_dir, seed=1337, ratio=(.7, .15, .15))



if __name__ == '__main__':
    # Split dataset
    """
    dir = "../../dataset/db_original"
    out_dir = "../../dataset/db_original_split"
    for f in os.listdir(dir):
        print(f)
    split_dataset_to_train_test_val(dir, out_dir)"""

    # Dataset to csv
    """
    dir = sys.argv[1]
    #dataset_info_to_csv(dir)"""


    dirr = "../../dataset/db_original_split"
    fs = ["train", "test", "val"]

    for f in fs:
        path = "{}/{}".format(dirr, f)
        classes = [name for name in os.listdir(path) if name != ".DS_Store"]
        print(classes)
        data = []

        tot = 0
        for label in classes:
            n = len([name for name in os.listdir("{}/{}".format(path, label)) if name.endswith("tiff")])
            data.append([label, n])
            tot += n
        print("\nfolder: ", f, "n: ", tot)
        print(data)

