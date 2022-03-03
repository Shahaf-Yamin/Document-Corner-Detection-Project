import random

import cv2
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import os
import yaml

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def intersection(a, b, img):
    img1 = np.zeros_like(img)

    cv2.fillConvexPoly(img1, a, (255, 0, 0))
    img1 = np.sum(img1, axis=2)

    img1 = img1 / 255

    img2 = np.zeros_like(img)
    cv2.fillConvexPoly(img2, b, (255, 0, 0))
    img2 = np.sum(img2, axis=2)
    img2 = img2 / 255

    inte = img1 * img2
    union = np.logical_or(img1, img2)
    iou = np.sum(inte) / np.sum(union)
    print(iou)
    return iou


def intersection_with_correction(a, b, img):
    img1 = np.zeros_like(img)
    cv2.fillConvexPoly(img1, a, (255, 0, 0))

    img2 = np.zeros_like(img)
    cv2.fillConvexPoly(img2, b, (255, 0, 0))
    min_x = min(a[0][0], a[1][0], a[2][0], a[3][0])
    min_y = min(a[0][1], a[1][1], a[2][1], a[3][1])
    max_x = max(a[0][0], a[1][0], a[2][0], a[3][0])
    max_y = max(a[0][1], a[1][1], a[2][1], a[3][1])

    dst = np.array(((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)))
    mat = cv2.getPerspectiveTransform(a.astype(np.float32), dst.astype(np.float32))
    img1 = cv2.warpPerspective(img1, mat, tuple((img.shape[0], img.shape[1])))
    img2 = cv2.warpPerspective(img2, mat, tuple((img.shape[0], img.shape[1])))

    img1 = np.sum(img1, axis=2)
    img1 = img1 / 255
    img2 = np.sum(img2, axis=2)
    img2 = img2 / 255

    inte = img1 * img2
    union = np.logical_or(img1, img2)
    iou = np.sum(inte) / np.sum(union)
    return iou

def intersection_with_correction_smart_doc_implementation(gt, prediction, img):

    # Reference : https://github.com/jchazalon/smartdoc15-ch1-eval

    gt = sort_gt(gt)
    prediction = sort_gt(prediction)
    img1 = np.zeros_like(img)
    cv2.fillConvexPoly(img1, gt, (255, 0, 0))

    target_width = 2100
    target_height = 2970
    # Referential: (0,0) at TL, x > 0 toward right and y > 0 toward bottom
    # Corner order: TL, BL, BR, TR
    # object_coord_target = np.float32([[0, 0], [0, target_height], [target_width, target_height], [target_width, 0]])
    object_coord_target = np.array(np.float32([[0, 0], [target_width, 0], [target_width, target_height],[0, target_height]]))
    # print (gt, object_coord_target)
    H = cv2.getPerspectiveTransform(gt.astype(np.float32).reshape(-1, 1, 2), object_coord_target.reshape(-1, 1, 2))

    # 2/ Apply to test result to project in target referential
    test_coords = cv2.perspectiveTransform(prediction.astype(np.float32).reshape(-1, 1, 2), H)

    # 3/ Compute intersection between target region and test result region
    # poly = Polygon.Polygon([(0,0),(1,0),(0,1)])
    poly_target = Polygon.Polygon(object_coord_target.reshape(-1, 2))
    poly_test = Polygon.Polygon(test_coords.reshape(-1, 2))
    poly_inter = poly_target & poly_test

    area_target = poly_target.area()
    area_test = poly_test.area()
    area_inter = poly_inter.area()

    area_union = area_test + area_target - area_inter
    # Little hack to cope with float precision issues when dealing with polygons:
    #   If intersection area is close enough to target area or GT area, but slighlty >,
    #   then fix it, assuming it is due to rounding issues.
    area_min = min(area_target, area_test)
    if area_min < area_inter and area_min * 1.0000000001 > area_inter:
        area_inter = area_min
        print("Capping area_inter.")

    jaccard_index = area_inter / area_union
    return jaccard_index



def __rotateImage(image, angle):
    rot_mat = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    result = cv2.warpAffine(image.astype(np.float32), rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return result, rot_mat

def plot_and_save(loss, path, name):
    plt.plot(list(range(len(loss))), loss)
    plt.title('Loss VS Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(path+name+'.png', dpi=512)
    with open(path+name+'.yml', 'w') as f:
        yaml.dump({'loss': loss}, f)


def rotate(img, gt, angle):
    img, mat = __rotateImage(img, angle)
    gt = gt.astype(np.float64)
    for a in range(0, 4):
        gt[a] = np.dot(mat[..., 0:2], gt[a]) + mat[..., 2]
    return img, gt


def random_crop(img, gt):
    ptr1 = (min(gt[0][0], gt[1][0], gt[2][0], gt[3][0]),
            min(gt[0][1], gt[1][1], gt[2][1], gt[3][1]))

    ptr2 = ((max(gt[0][0], gt[1][0], gt[2][0], gt[3][0]),
             max(gt[0][1], gt[1][1], gt[2][1], gt[3][1])))

    start_x = np.random.randint(0, int(max(ptr1[0] - 1, 1)))
    start_y = np.random.randint(0, int(max(ptr1[1] - 1, 1)))

    end_x = np.random.randint(int(min(ptr2[0] + 1, img.shape[1] - 1)), img.shape[1])
    end_y = np.random.randint(int(min(ptr2[1] + 1, img.shape[0] - 1)), img.shape[0])

    img = img[start_y:end_y, start_x:end_x]

    myGt = gt - (start_x, start_y)
    myGt = myGt * (1.0 / img.shape[1], 1.0 / img.shape[0])

    myGtTemp = myGt * myGt
    sum_array = myGtTemp.sum(axis=1)
    tl_index = np.argmin(sum_array)
    tl = myGt[tl_index]
    tr = myGt[(tl_index + 1) % 4]
    br = myGt[(tl_index + 2) % 4]
    bl = myGt[(tl_index + 3) % 4]

    return img, (tl, tr, br, bl)


def get_corners(img, gt):
    gt = gt.astype(int)
    list_of_points = {}
    myGt = gt

    myGtTemp = myGt * myGt
    sum_array = myGtTemp.sum(axis=1)

    tl_index = np.argmin(sum_array)
    tl = myGt[tl_index]
    tr = myGt[(tl_index + 1) % 4]
    br = myGt[(tl_index + 2) % 4]
    bl = myGt[(tl_index + 3) % 4]

    list_of_points["tr"] = tr
    list_of_points["tl"] = tl
    list_of_points["br"] = br
    list_of_points["bl"] = bl
    gt_list = []
    images_list = []
    for k, v in list_of_points.items():

        if (k == "tl"):
            cords_x = __get_cords(v[0], 0, list_of_points["tr"][0], buf=10, size=abs(list_of_points["tr"][0] - v[0]))
            cords_y = __get_cords(v[1], 0, list_of_points["bl"][1], buf=10, size=abs(list_of_points["bl"][1] - v[1]))
            # print cords_y, cords_x
            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

            cut_image = img[cords_y[0]:cords_y[1], cords_x[0]:cords_x[1]]

        if (k == "tr"):
            cords_x = __get_cords(v[0], list_of_points["tl"][0], img.shape[1], buf=10,
                                  size=abs(list_of_points["tl"][0] - v[0]))
            cords_y = __get_cords(v[1], 0, list_of_points["br"][1], buf=10, size=abs(list_of_points["br"][1] - v[1]))
            # print cords_y, cords_x
            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

            cut_image = img[cords_y[0]:cords_y[1], cords_x[0]:cords_x[1]]

        if (k == "bl"):
            cords_x = __get_cords(v[0], 0, list_of_points["br"][0], buf=10,
                                  size=abs(list_of_points["br"][0] - v[0]))
            cords_y = __get_cords(v[1], list_of_points["tl"][1], img.shape[0], buf=10,
                                  size=abs(list_of_points["tl"][1] - v[1]))
            # print cords_y, cords_x
            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

            cut_image = img[cords_y[0]:cords_y[1], cords_x[0]:cords_x[1]]

        if (k == "br"):
            cords_x = __get_cords(v[0], list_of_points["bl"][0], img.shape[1], buf=10,
                                  size=abs(list_of_points["bl"][0] - v[0]))
            cords_y = __get_cords(v[1], list_of_points["tr"][1], img.shape[0], buf=10,
                                  size=abs(list_of_points["tr"][1] - v[1]))
            # print cords_y, cords_x
            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

            cut_image = img[cords_y[0]:cords_y[1], cords_x[0]:cords_x[1]]

        # cv2.circle(cut_image, gt, 2, (255, 0, 0), 6)
        mah_size = cut_image.shape
        cut_image = cv2.resize(cut_image, (300, 300))
        a = int(gt[0] * 300 / mah_size[1])
        b = int(gt[1] * 300 / mah_size[0])
        images_list.append(cut_image)
        gt_list.append((a, b))
    return images_list, gt_list


def __get_cords(cord, min_start, max_end, size=299, buf=5, random_scale=True):
    # size = max(abs(cord-min_start), abs(cord-max_end))
    iter = 0
    if (random_scale):
        size /= random.randint(1, 4)
    while (max_end - min_start) < size:
        size = size * .9
    temp = -1
    while (temp < 1):
        temp = random.normalvariate(size / 2, size / 6)
    x_start = max(cord - temp, min_start)
    x_start = int(x_start)
    if x_start >= cord:
        print("XSTART AND CORD", x_start, cord)
    assert (x_start < cord)
    while ((x_start < min_start) or (x_start + size > max_end) or (x_start + size <= cord)):
        # x_start = random.randint(int(min(max(min_start, int(cord - size + buf)), cord - buf - 1)), cord - buf)
        temp = -1
        while (temp < 1):
            temp = random.normalvariate(size / 2, size / 6)
        temp = max(temp, 1)
        x_start = max(cord - temp, min_start)
        x_start = int(x_start)
        size = size * .995
        iter += 1
        if (iter == 1000):
            x_start = int(cord - (size / 2))
            print("Gets here")
            break
    assert (x_start >= 0)
    if x_start >= cord:
        print("XSTART AND CORD", x_start, cord)
    assert (x_start < cord)
    assert (x_start + size <= max_end)
    assert (x_start + size > cord)
    return (x_start, int(x_start + size))


def setup_logger(path):
    import logging
    logger = logging.getLogger('iCARL')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(path + ".log")
    fh.setLevel(logging.DEBUG)

    fh2 = logging.FileHandler("../temp.log")
    fh2.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    fh2.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(fh2)
    logger.addHandler(ch)
    return logger


def sort_gt(gt):
    '''
    Sort the ground truth labels so that TL corresponds to the label with smallest distance from O
    :param gt: 
    :return: sorted gt
    '''
    myGtTemp = gt * gt
    sum_array = myGtTemp.sum(axis=1)
    tl_index = np.argmin(sum_array)
    tl = gt[tl_index]
    tr = gt[(tl_index + 1) % 4]
    br = gt[(tl_index + 2) % 4]
    bl = gt[(tl_index + 3) % 4]

    return np.asarray((tl, tr, br, bl))
