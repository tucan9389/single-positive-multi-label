import os
import json
import numpy as np
import argparse
import shutil
from tqdm import tqdm
from PIL import Image

pp = argparse.ArgumentParser(description='Format PASCAL 2012 metadata.')
pp.add_argument('--load-path', type=str, default='../data/pascal_food', help='Path to a directory containing a copy of the PASCAL dataset.')
pp.add_argument('--save-path', type=str, default='../data/pascal_food', help='Path to output directory.')
args = pp.parse_args()

food_dataset_path = '/workspace/storage/cephfs-pai/model/application/vision/food_classification/train/datasets_v4_real/'

dataset_path = os.path.join(food_dataset_path, 'train')
subcategory_nos = os.listdir(dataset_path)
subcategory_nos.sort()
catName_to_catID = dict(list(zip(subcategory_nos, list(range(len(subcategory_nos))))))

print(catName_to_catID)

# catName_to_catID = {
#     'aeroplane': 0,
#     'bicycle': 1,
#     'bird': 2,
#     'boat': 3,
#     'bottle': 4,
#     'bus': 5,
#     'car': 6,
#     'cat': 7,
#     'chair': 8,
#     'cow': 9,
#     'diningtable': 10,
#     'dog': 11,
#     'horse': 12,
#     'motorbike': 13,
#     'person': 14,
#     'pottedplant': 15,
#     'sheep': 16,
#     'sofa': 17,
#     'train': 18,
#     'tvmonitor': 19
# }

catID_to_catName = {catName_to_catID[k]: k for k in catName_to_catID}

total_img_names = []
dst_img_dir_path = os.path.join(args.load_path, 'food_v4', 'JPEGImages')
dst_annotation_path = os.path.join(args.load_path, 'food_v4', 'ImageSets', 'Main')
os.makedirs(dst_img_dir_path, exist_ok=True)
os.makedirs(dst_annotation_path, exist_ok=True)


# copy images
for phase in ['train', 'val']:
    dir_name = 'train'
    if phase == 'val':
        dir_name = 'test'
    dataset_path = os.path.join(food_dataset_path, dir_name)
    
    for i, cat in enumerate(subcategory_nos):
        print(f"START {i}th/{len(subcategory_nos)} cat: {cat}")
        cat_path = os.path.join(dataset_path, cat)
        img_names = os.listdir(cat_path)

        for img_name in tqdm(img_names):
            src_img_path = os.path.join(cat_path, img_name)
            tmp = img_name.split(".")
            if len(tmp) != 2:
                continue

            old_img_name = img_name
            img_name = img_name.replace("\\", "")
            img_name = img_name.replace(" ", "")
            img_name = img_name.replace("(", "_")
            img_name = img_name.replace(")", "_")
            filename = img_name.split(".")[0]
            extname = img_name.split(".")[1]
            dst_img_name = cat + '__' + filename + '.jpg'
            dst_img_path = os.path.join(dst_img_dir_path, dst_img_name)

            # if '247_1000063' in dst_img_name:
            #     print('old_img_name:', old_img_name)
            #     print('filename:', filename)
            #     print('extname:', extname)
            #     print('img_name:', img_name)
            #     print('dst_img_name:', dst_img_name)
            #     exit(0)
            # continue

            old_src_img_path = os.path.join(dst_img_dir_path, old_img_name.split(".")[0] + '.jpg')
            if os.path.exists(old_src_img_path):
                # if '247_1000063' in dst_img_name:
                #     print("0 rename:", old_src_img_path, dst_img_path)
                os.rename(old_src_img_path, dst_img_path)
            elif not os.path.exists(dst_img_path):
                if extname == 'jpg':
                    # if '247_1000063' in dst_img_name:
                    #     print("1 shutil.copyfile:", src_img_path, dst_img_path)
                    shutil.copyfile(src_img_path, dst_img_path)
                elif extname.lower() == 'jpg':
                    # if '247_1000063' in dst_img_name:
                    #     print("2 shutil.copyfile:", src_img_path, dst_img_path)
                    dst_img_name = cat + '__' + filename + '.' + extname.lower()
                    dst_img_path = os.path.join(dst_img_dir_path, dst_img_name)
                    shutil.copyfile(src_img_path, dst_img_path)
                else:
                    # if '247_1000063' in dst_img_name:
                    #     print("3 img.save:", src_img_path, dst_img_path)
                    dst_img_name = cat + '__' + filename + '.jpg'
                    dst_img_path = os.path.join(dst_img_dir_path, dst_img_name)
                    img = Image.open(src_img_path)
                    img = img.convert('RGB')
                    img.save(dst_img_path)
            else:
                # if '247_1000063' in dst_img_name:
                #     print("4 pass:", src_img_path, dst_img_path)
                pass

            total_img_names.append(dst_img_name)

# exit(0)

# make pascal annotation files
for phase in ['train', 'val']:
    dir_name = 'train'
    if phase == 'val':
        dir_name = 'test'
    dataset_path = os.path.join(food_dataset_path, dir_name)

    print()
    print(f"make pascal annotation files - {phase}")
    for i, cat in tqdm(list(enumerate(subcategory_nos))):
        cat_path = os.path.join(dataset_path, cat)
        img_names = os.listdir(cat_path)
        img_names = [img_name.replace("\\", "") for img_name in img_names]
        img_names = [img_name.replace(" ", "") for img_name in img_names]
        img_names = [img_name.replace("(", "_") for img_name in img_names]
        img_names = [img_name.replace(")", "_") for img_name in img_names]
        target_img_names = [cat + '__' + img_name for img_name in img_names]
        target_img_names = set(target_img_names)

        annotation_filename = cat + '_' + phase + '.txt'
        annotation_file_path = os.path.join(dst_annotation_path, annotation_filename)
        if os.path.exists(annotation_file_path):
            continue
        
        annotation_text = ''
        for img_name in total_img_names:
            if len(img_name.split(".")) != 2:
                continue
            if img_name in target_img_names:
                annotation_text += img_name.split(".")[0] + '  1' + '\n'
            else:
                annotation_text += img_name.split(".")[0] + ' -1' + '\n'
        f = open(annotation_file_path, 'w')
        f.write(annotation_text)


# make single-positive annotation format
ann_dict = {}
image_list = {'train': [], 'val': []}

for phase in ['train', 'val']:
    print()
    print(f"make single-positive annotation format - {phase}")
    for cat in tqdm(catName_to_catID):
        annotation_filename = cat + '_' + phase + '.txt'
        with open(os.path.join(dst_annotation_path, annotation_filename), 'r') as f:
            for line in f:
                cur_line = line.rstrip().split(' ')
                image_id = cur_line[0]
                label = cur_line[-1]
                image_fname = image_id + '.jpg'
                if int(label) == 1:
                    if image_fname not in ann_dict:
                        ann_dict[image_fname] = []
                        image_list[phase].append(image_fname)
                    ann_dict[image_fname].append(catName_to_catID[cat])
    # create label matrix: 
    image_list[phase].sort()
    num_images = len(image_list[phase])
    label_matrix = np.zeros((num_images, len(catName_to_catID)))
    for i in range(num_images):
        cur_image = image_list[phase][i]
        label_indices = np.array(ann_dict[cur_image])
        label_matrix[i, label_indices] = 1.0
    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_labels.npy'), label_matrix)
    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_images.npy'), np.array(image_list[phase]))
