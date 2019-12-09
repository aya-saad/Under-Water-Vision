
import tensorflow as tf
import numpy as np
import os
import pathlib
import cv2
import split_folders

AUTOTUNE = tf.data.experimental.AUTOTUNE


def rescale_image(filepath, filepath_out):
    """
    img = Image.open(filepath)
    org_size = img.size
    ratio = float(IMG_SIZE) / max(org_size)
    new_size = tuple([int(x * ratio) for x in org_size])

    # BILINEAR, NEAREST, BICUBIC, ANTIALIAS
    #options = {"Bilinear": Image.BILINEAR, "Nearest": Image.NEAREST, "Bicubinc": Image.BICUBIC, "Antialias": Image.ANTIALIAS}
    #for name in options:
    img_resized = img.resize(new_size, Image.ANTIALIAS)
    #new_im = Image.new("L", (IMG_SIZE, IMG_SIZE))   # L is for black and white
    #new_im.paste(img_resized, ((IMG_SIZE - new_size[0]) // 2, (IMG_SIZE - new_size[1]) // 2))

    delta_w = IMG_SIZE - new_size[0]
    delta_h = IMG_SIZE - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    new_im = ImageOps.expand(img_resized, padding)
    greyscale_img = ImageOps.grayscale(new_im)

    greyscale_img.show()
    #img_resized.save("{}_{}".format(name, filepath_out))
"""
    desired_size = IMG_SIZE
    im_pth = filepath

    im = cv2.imread(im_pth)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REPLICATE)
    cv2.imwrite(filepath_out, new_im)
    print(filepath_out)

    return new_im

def rescale_corpus(dir, classes):
    for img_class in classes:
        img_class = img_class.lower()
        img_class_path = "{}/{}".format(dir, img_class)
        img_dir_out = "{}_scaled32".format(corpus)
        img_class_path_out = "{}_scaled32/{}".format(corpus, img_class)

        if not os.path.exists(img_class_path_out):
            os.makedirs(img_class_path_out)

        n = 0
        for img in os.listdir(img_class_path):
            if n < 2000:
                if img.split(".")[-1] == "png":
                    filepath = "{}/{}".format(img_class_path, img)
                    filepath_out = "{}/{}".format(img_class_path_out, img)
                    rescale_image(filepath, filepath_out)
                    n += 1
    return img_dir_out


def load_corpus(data_dir):
    data_dir = pathlib.Path(data_dir)
    class_names = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])
    print(class_names)
    image_count = len(list(data_dir.glob('*/*.png')))
    print("Image count: {}".format(image_count))
    list_ds = tf.data.Dataset.list_files("{}/*/*".format(data_dir))
    return list_ds, class_names, image_count


if __name__ == '__main__':
    IMG_SIZE = 32

    classes = ["detritus", "Leptocylindrus", "Rhizosolenia",
               "Cylindrotheca", "dino30"]
    corpus = "2014"

    scaled_corpus = rescale_corpus(corpus, classes)

    scaled_corpus_split = "{}_split".format(scaled_corpus)
    split_folders.ratio(scaled_corpus, output=scaled_corpus_split, seed=1337, ratio=(.9, .1))  # default values












