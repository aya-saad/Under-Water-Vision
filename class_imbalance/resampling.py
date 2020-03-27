import sys
import os
from PIL import Image
import random
import pandas as pd

BIGGEST_CLASS = "bubble"
SMALLEST_CLASS = "fish_egg"

def get_number_of_instances(path):
    class_instances = {}
    species = [spec for spec in os.listdir(path) if os.path.isdir(os.path.join(path, spec))]
    print(species)
    for spec in species:
        species_path = "{}/{}".format(path, spec)
        imgs = [img for img in os.listdir(species_path) if img.endswith("tiff")]
        n_imgs = len(imgs)
        class_instances[spec] = n_imgs

    return class_instances

def path_exists(species_path, img_path):
    full_path = "{}/{}".format(species_path, img_path)
    if os.path.exists(full_path):
        first, second, third = img_path.split(".")[0], img_path.split(".")[1], img_path.split(".")[2]
        add = "copy"
        new_img_path = "{}{}.{}.{}".format(first, add, second, third)
        return path_exists(species_path, new_img_path)
    else:
        return str(full_path)

def random_minority_oversampling(path, class_instances):
    for species in class_instances:
        species_path = "{}/{}".format(path, species)
        imgs = [img for img in os.listdir(species_path) if img.endswith("tiff")]

        n_imgs = class_instances[species]
        n_max = class_instances[BIGGEST_CLASS]
        n_copies = n_max - n_imgs
        print("label: ", species, "n_copies: ", n_copies)

        for i in range(n_copies):
            n = random.randint(0, n_imgs-1)
            img_path = "{}/{}".format(species_path, imgs[n])
            save_path = path_exists(species_path, imgs[n])
            print("random number: ", n, "save path: ", save_path)
            im1 = Image.open(img_path)
            im1.save(save_path)

def random_undersampling(path, class_instances):
    for species in class_instances:
        species_path = "{}/{}".format(path, species)
        imgs = [img for img in os.listdir(species_path) if img.endswith("tiff")]

        n_imgs = class_instances[species]
        n_max = class_instances[SMALLEST_CLASS]
        n_removals = n_imgs - n_max
        print("label: ", species, "n_removals: ", n_removals)

        for i in range(n_removals):
            n = random.randint(0, len(imgs) - 1)
            os.remove(os.path.join(path, species, imgs[n]))
            imgs.pop(n)





if __name__ == '__main__':
    #path = sys.argv[0]
    path_original = "../../dataset/db_undersampled/train"
    class_instances = get_number_of_instances(path_original)
    print("original ", class_instances)


    path_undersampled = "../../dataset/db_undersampled/train"
    class_instances = get_number_of_instances(path_undersampled)
    #random_undersampling(path_undersampled, class_instances)
    print("oversampled ", class_instances)

    #random_minority_oversampling(path, class_instances)

