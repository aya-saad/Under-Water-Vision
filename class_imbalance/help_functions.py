import os
import sys
import cv2
import split_folders
import skimage
from skimage import io, transform
import numpy as np
import pandas as pd



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

def rescale_dataset(filepath, savepath):
    print(filepath)
    classes = [c for c in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, c))]
    print(classes)
    for c in classes:
        class_path = os.path.join(filepath, c)
        files = [o for o in os.listdir(class_path) if o.endswith('.tiff')]
        print(class_path, len(files))
        save_path = os.path.join(savepath, c)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for f in files:
            img = skimage.io.imread(os.path.join(class_path, f), plugin="tifffile")
            img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            img = img.astype('float32')
            try:
                img = cv2.medianBlur(blur_limit=3)(image=img).get('image')
            except Exception:
                fail = 1
            img = cv2.medianBlur(img, 3)
            img = skimage.transform.resize(img, (64, 64, 3), mode='reflect', preserve_range=True)
            img = img.astype(np.uint8)
            skimage.io.imsave(os.path.join(save_path, f), arr=img, plugin="tifffile")



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

    # Rescale dataset
    """filepath = "../../dataset/db_original/train"
    savepath = "../../dataset/db_original_scaled/train"
    rescale_dataset(filepath, savepath)
    filepath = "../../dataset/db_original/test"
    savepath = "../../dataset/db_original_scaled/test"
    rescale_dataset(filepath, savepath)
    filepath = "../../dataset/db_original/val"
    savepath = "../../dataset/db_original_scaled/val"
    rescale_dataset(filepath, savepath)"""


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

