import sys
import os
from PIL import Image
import random
import pandas as pd

N_MAX = 2636

class_instances = {
    "copepod":          657,
    "faecal_pellets":   504,
    "other":            1931,
    "fish_egg":         213,
    "bubble":           2636,
    "diatom_chain":     850,
    "oil":              671,
    "oily_gas":         479
}

def random_minority_oversampling(path):
    for species in class_instances:
        species_path = "{}/{}".format(path, species)
        imgs = [img for img in os.listdir(species_path) if img.endswith("tiff")]
        n_imgs = len(imgs)
        n_copies = N_MAX - n_imgs
        for i in range(n_copies):
            n = random.randint(0, n_imgs-1)
            img_path = "{}/{}".format(species_path, imgs[n])
            save_path = path_exists(species_path, imgs[n])
            print(save_path)
            im1 = Image.open(img_path)
            im1.save(save_path)



        """
        for img in imgs:
            img_path = "{}/{}".format(species_path, img)
            print(img_path)
            im1 = Image.open(img_path)
            im2 = im1.copy()"""


def path_exists(species_path, img_path):
    full_path = "{}/{}".format(species_path, img_path)
    if os.path.exists(full_path):
        first, second, third = img_path.split(".")[0], img_path.split(".")[1], img_path.split(".")[2]
        add = "copy"
        new_img_path = "{}{}.{}.{}".format(first, add, second, third)
        return path_exists(species_path, new_img_path)
    else:
        return str(full_path)


def dataset_info_to_csv(dir):
    classes = [name for name in os.listdir(dir) if name != ".DS_Store"]
    print(classes)
    data = []

    for label in classes:
        n = len([name for name in os.listdir("{}/{}".format(dir, label)) if
                 name.endswith(".tiff")])
        data.append([label, n])

    df = pd.DataFrame(data, columns=['Label', 'n_img'])
    df = df.append(df.sum(numeric_only=True), ignore_index=True)
    print(df)

    df.to_csv("{}_info.csv".format(dir.split("/")[0]))

if __name__ == '__main__':
    #path = sys.argv[0]
    path = "../../dataset_oda_oversampled"
    #random_minority_oversampling(path)
    dataset_info_to_csv(path)

