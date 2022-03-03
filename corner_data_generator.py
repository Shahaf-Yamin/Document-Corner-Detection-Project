import os
import cv2
import numpy as np
from utils import utils, data_loading, old_utils
from tqdm import tqdm


def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="Path to data files (Extract images using video_to_image.py first")
    parser.add_argument("-o", "--output-dir", help="Directory to store results")
    parser.add_argument("--dataset", default="smartdoc", help="'smartdoc' or 'selfcollected' dataset")
    return parser.parse_args()


if __name__ == '__main__':
    args = args_processor()
    input_directory = args.input_dir
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    import csv

    dataset = utils.SmartDocDirectories(directory=input_directory)
    with open(os.path.join(args.output_dir, 'gt.csv'), 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Counter for file naming
        counter = 0
        for data_elem in tqdm(dataset.myData):
            img_path = data_elem[0]
            target = data_elem[1].reshape((4, 2))
            img = cv2.imread(img_path)
            corner_cords = target
            mask = img #utils.gen_mask(coordinates=corner_cords, sizeX=img.shape[0], sizeY=img.shape[1], sort=True)
            for angle in range(0, 1, 90):
                mask_rotate, gt_rotate = old_utils.rotate(mask, corner_cords, angle)
                for random_crop in range(0, 1):
                    mask_list, gt_list = old_utils.get_corners(mask_rotate, gt_rotate)
                    for a in range(0, 4):
                        counter += 1
                        f_name = str(counter).zfill(8)
                        gt_store = list(np.array(gt_list[a]) / (300, 300))
                        mask_store = cv2.resize(255*mask_list[a], (64, 64))
                        cv2.imwrite(os.path.join(args.output_dir, f_name + ".png"), mask_store)
                        spamwriter.writerow((f_name + ".png", tuple(gt_store)))
