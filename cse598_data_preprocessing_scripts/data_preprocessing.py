import os
import tqdm
import json
import numpy as np
from PIL import Image
from class_id import class_ids

# Disabling DecompressionBomb safeguard
Image.MAX_IMAGE_PIXELS = None

def filter_by_suffix(file_all, suffix):
    return [f for f in file_all if f.endswith(suffix)]


def resize_and_pad(img, size=(224, 224), pad_color=0):
    # Determine the original size of the image
    original_size = img.size
    # Find the ratio needed to resize the image to have its longer side be 224
    ratio = float(size[0]) / max(original_size)
    new_size = tuple([int(x * ratio) for x in original_size])
    # Resize the image
    img = img.resize(new_size)
    # Create a new image with the specified size and black background
    new_img = Image.new("RGB", size, pad_color)
    # Get the coordinates to paste the resized image onto the center of the background
    upper_left = ((size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2)
    new_img.paste(img, upper_left)
    return new_img

# ----- #
# config region
# ----- #
dataset_type = "train"
path_dataset = "/scratch/ccui17/fmow_{}".format(dataset_type)
path_x = "/scratch/ccui17/images_{}.memmap".format(dataset_type)
path_y = "/scratch/ccui17/labels_{}.memmap".format(dataset_type)
if dataset_type == "train":
    num_sample = 363572
if dataset_type == "val":
    num_sample = 53041
if dataset_type == "test":
    raise Exception()

f_images = np.memmap(path_x, dtype="uint8", mode="w+", shape=(num_sample, 224, 224, 3))
f_labels = np.memmap(path_y, dtype="int64", mode="w+", shape=(num_sample,))

# mapper makes sure the dataset is shuffled
mapper = np.arange(num_sample)
np.random.shuffle(mapper)

counter_sample = 0
class_all = os.listdir(path_dataset)
for idx_class in range(len(class_all)):
    class_name = class_all[idx_class]
    path_class_folder = os.path.join(path_dataset, class_name)
    location_all = os.listdir(path_class_folder)
    for idx_location in tqdm.tqdm(range(len(location_all))):
        location_name = location_all[idx_location]
        path_location_folder = os.path.join(path_class_folder, location_name)
        sample_all = os.listdir(path_location_folder)
        sample_all = filter_by_suffix(sample_all, "_rgb.jpg")
        for idx_sample in range(len(sample_all)):       
            sample_name_jpg = sample_all[idx_sample]
            sample_name_json = sample_name_jpg[:-4] + ".json"
            path_sample_jpg = os.path.join(path_location_folder, sample_name_jpg)
            path_sample_json = os.path.join(path_location_folder, sample_name_json)

            img = Image.open(path_sample_jpg)
            with open(path_sample_json, "r") as f:
                json_content = json.load(f)

            X, Y, W, H = json_content["bounding_boxes"][0]["box"]
            img = img.crop((X, Y, X + W, Y + H))
            img = resize_and_pad(img).convert("RGB")

            f_images[mapper[counter_sample]] = np.array(img, dtype="uint8")
            f_labels[mapper[counter_sample]] = class_ids[json_content["bounding_boxes"][0]["category"]]
            counter_sample += 1

# flush and close the memmap files
f_images.flush()
f_labels.flush()
del f_images
del f_labels
