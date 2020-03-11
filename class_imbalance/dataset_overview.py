import os
import sys
import pandas as pd


def dataset_info_to_csv(dir):
    classes = [name for name in os.listdir(dir) if name != ".DS_Store"]
    data = []

    for label in classes:
        n = len([name for name in os.listdir("{}/{}".format(dir, label)) if
                 os.path.isfile(os.path.join(dir, label, name))])
        data.append([label, n])

    df = pd.DataFrame(data, columns=['Label', 'n_img'])
    print(df)

    df.to_csv("{}_info.csv".format(dir))


if __name__ == '__main__':
    dir = sys.argv[0]
    dataset_info_to_csv(dir)
