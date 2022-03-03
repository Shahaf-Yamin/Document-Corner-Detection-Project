import tqdm
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
from PIL import Image
from matplotlib.path import Path
import xml.etree.ElementTree as ET
# import skimage
# from skimage import measure


def sort_predication(masks_pred):
    sort_idx = np.argsort(masks_pred[:, 0])
    tl_idx = np.argmin(masks_pred[sort_idx][:2][:, 1])
    tr_idx = np.argmax(masks_pred[sort_idx][:2][:, 1])
    tl = masks_pred[sort_idx][:2][tl_idx]
    tr = masks_pred[sort_idx][:2][tr_idx]
    bl_idx = np.argmin(masks_pred[sort_idx][2:][:, 1])
    br_idx = np.argmax(masks_pred[sort_idx][2:][:, 1])
    bl = masks_pred[sort_idx][2:][bl_idx]
    br = masks_pred[sort_idx][2:][br_idx]
    return np.array([tl, tr, br, bl])

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

def gen_mask(coordinates, sizeX, sizeY, sort):
    nx, ny = sizeX, sizeY
    if sort:
        coordinates = sort_predication(coordinates)
        coordinates = list(map(tuple, coordinates))

    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    coordinates.append((0., 0.))
    poly_verts = coordinates
    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    y, x = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(poly_verts, codes=codes)
    grid = path.contains_points(points)
    mask = grid.reshape((ny, nx)).T
    # if sort:
    #     plt.figure(figsize=[10.8, 19.2])
    #     plt.imshow(mask)
    #     plt.scatter([x for x, y in coordinates], [y for x,y in coordinates])
    #     plt.show()
    return mask

# def clean_predicated_mask_artifacts(input_mask):
#     labels_mask = measure.label(input_mask)
#     regions = measure.regionprops(labels_mask)
#     regions.sort(key=lambda x: x.area, reverse=True)
#     if len(regions) > 1:
#         for rg in regions[1:]:
#             labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
#     labels_mask[labels_mask != 0] = 1
#     mask = labels_mask
#     return mask

class _Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, name):
        self.name = name
        self.data = []
        self.labels = []

class SmartDocDirectories(_Dataset):
    '''
    Class to include SmartDoc Dataset via full resolution images
    '''

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []

        for folder in os.listdir(directory):
            if (os.path.isdir(directory + "/" + folder)):
                for file in os.listdir(directory + "/" + folder):
                    images_dir = directory + "/" + folder + "/" + file
                    if (os.path.isdir(images_dir)):

                        list_gt = []
                        tree = ET.parse(images_dir + "/" + file + ".gt")
                        root = tree.getroot()
                        for a in root.iter("frame"):
                            list_gt.append(a)

                        im_no = 0
                        for image in os.listdir(images_dir):
                            if image.endswith(".jpg"):
                                im_no += 1

                                # Now we have opened the file and GT. Write code to create multiple files and scale gt
                                list_of_points = {}

                                self.data.append(os.path.join(images_dir, image))
                                imageIndex = int(float(image[0:-4])) - 1
                                for point in list_gt[imageIndex].iter("point"):
                                    myDict = point.attrib

                                    list_of_points[myDict["name"]] = (int(float(myDict['x'])), int(float(myDict['y'])))

                                ground_truth = np.asarray((list_of_points["tl"], list_of_points["tr"], list_of_points["br"], list_of_points["bl"]))
                                ground_truth = sort_gt(ground_truth)
                                self.labels.append(ground_truth)

        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 8))
        print("Ground Truth Shape: %s", str(self.labels.shape))
        print("Data shape %s", str(len(self.data)))

        self.myData = []
        for a in range(len(self.data)):
            self.myData.append([self.data[a], self.labels[a]])

class PartialRandomSmartDocDirectories(_Dataset):
    '''
    Class to include SmartDoc Dataset via full resolution images while sampling only 2 images from a single video
    '''

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []

        for folder in os.listdir(directory):
            if (os.path.isdir(directory + "/" + folder)):
                for file in os.listdir(directory + "/" + folder):
                    images_dir = directory + "/" + folder + "/" + file
                    if (os.path.isdir(images_dir)):

                        list_gt = []
                        tree = ET.parse(images_dir + "/" + file + ".gt")
                        root = tree.getroot()
                        for a in root.iter("frame"):
                            list_gt.append(a)

                        im_no = 0
                        # List all available images in a list
                        images_list = os.listdir(images_dir)
                        # Remove from the list the Ground True
                        images_list.remove(file + ".gt")
                        # if 'background05' in images_dir:
                        #     # ChosenImages = list(np.random.choice(os.listdir(images_dir), size=len(os.listdir(images_dir))))
                        #     ChosenImages = list(np.random.choice(os.listdir(images_dir), size=1))
                        # else:
                        #     # ChosenImages = list(np.random.choice(os.listdir(images_dir), size=len(os.listdir(images_dir))))
                        #     ChosenImages = list(np.random.choice(os.listdir(images_dir), size=1))
                        ChosenImages = list(np.random.choice(os.listdir(images_dir), size=1))
                        for image in ChosenImages:
                            if image.endswith(".jpg"):
                                im_no += 1

                                # Now we have opened the file and GT. Write code to create multiple files and scale gt
                                list_of_points = {}

                                self.data.append(os.path.join(images_dir, image))
                                imageIndex = int(float(image[0:-4])) - 1
                                for point in list_gt[imageIndex].iter("point"):
                                    myDict = point.attrib

                                    list_of_points[myDict["name"]] = (int(float(myDict['x'])), int(float(myDict['y'])))

                                ground_truth = np.asarray((list_of_points["tl"], list_of_points["tr"], list_of_points["br"], list_of_points["bl"]))
                                ground_truth = sort_gt(ground_truth)
                                self.labels.append(ground_truth)

        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 8))
        print("Ground Truth Shape: %s", str(self.labels.shape))
        print("Data shape %s", str(len(self.data)))

        self.myData = []
        for a in range(len(self.data)):
            self.myData.append([self.data[a], self.labels[a]])

class SmartDoc(_Dataset):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []
        for d in directory:
            self.directory = d
            self.classes_list = {}

            file_names = []
            print (self.directory, "gt.csv")
            with open(os.path.join(self.directory, "gt.csv"), 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                import ast
                for row in spamreader:
                    file_names.append(row[0])
                    self.data.append(os.path.join(self.directory, row[0]))
                    test = row[1].replace("array", "")
                    self.labels.append((ast.literal_eval(test)))
        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 8))
        self.myData = [self.data, self.labels]

class SmartDocCorner(_Dataset):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self, directory="data"):
        super().__init__("smartcorner")
        for d in directory:
            self.directory = d
            self.classes_list = {}

            file_names = []
            with open(os.path.join(self.directory, "gt.csv"), 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                import ast
                for row in spamreader:
                    file_names.append(row[0])
                    self.data.append(os.path.join(self.directory, row[0]))
                    test = row[1].replace("array", "")
                    self.labels.append((ast.literal_eval(test)))
        self.labels = np.array(self.labels)

        self.labels = np.reshape(self.labels, (-1, 2))
        self.myData = [self.data, self.labels]

def generate_full_resolution_dataset():
    dataset = SmartDocDirectories(directory='data/256TrainInference')
    images = []
    labels = []
    masks = []
    NEW_SIZE = 256
    for img_path, label in tqdm.tqdm(dataset.myData[:][-10:]):
        img = Image.open(img_path).resize((NEW_SIZE, NEW_SIZE))
        img = np.array(img).astype(np.float) / 256
        # Extract the x and y coordinates
        # x_cords = label[[0, 2, 4, 6]]
        # y_cords = label[[1, 3, 5, 7]] * 1072 / 1080
        x_cords = label[[0, 2, 4, 6]] * NEW_SIZE / 1920
        y_cords = label[[1, 3, 5, 7]] * NEW_SIZE / 1080
        labels.append([(i, j) for i, j in zip(x_cords, y_cords)])
        mask = gen_mask(labels[-1], sizeX=img.shape[0], sizeY=img.shape[1], sort=False)
        #TODO: remove transposee in predication
        images.append(img.T)
        masks.append(mask.T)
    return images, masks, labels

def generate_full_resolution_partial_dataset(path, size):
    dataset = PartialRandomSmartDocDirectories(directory=path)
    images = []
    labels = []
    masks = []
    X_Indices = [0, 2, 4, 6]
    Y_Indices = [1, 3, 5, 7]
    for img_path, label in tqdm.tqdm(dataset.myData):
        img = Image.open(img_path)
        old_size = img.size
        x_cords = label[X_Indices] * (size / old_size[0])
        y_cords = label[Y_Indices] * (size / old_size[1])
        img = img.resize((size, size))
        img = np.array(img) #/ 255
        currLabel = [(i, j) for i, j in zip(x_cords, y_cords)]
        normLabel = [(float(x) /old_size[0], float(y)/old_size[1]) for x,y in zip(label[X_Indices],label[Y_Indices])]
        normLabel = list(sum(normLabel,()))
        mask = gen_mask(currLabel, sizeX=img.shape[0], sizeY=img.shape[1], sort=False)
        labels.append(normLabel)
        images.append(img.T)
        masks.append(mask.T)
    stats = {'mean': np.mean(np.array(images)), 'std': np.std(np.array(images))}

    return images, masks, labels, stats

def LoadMaskDocumentDataset(path):
    dataset = SmartDoc(directory=[path])
    labels = []
    masks = []
    for mask_path, norm_label in tqdm.tqdm(zip(dataset.myData[0], dataset.myData[1])):
        mask = np.array(Image.open(mask_path))
        labels.append(norm_label)
        masks.append(np.expand_dims(mask,axis=0))
    return masks, labels

def LoadMaskCornerDataset(path):
    dataset = SmartDocCorner(directory=[path])
    labels = []
    masks = []
    for mask_path, norm_label in tqdm.tqdm(zip(dataset.myData[0], dataset.myData[1])):
        mask = np.array(Image.open(mask_path))
        labels.append(norm_label)
        masks.append(np.expand_dims(mask, axis=0))
    return masks, labels

def generate_dataset(path):
    # dataset = SmartDoc(directory=['data/imgs'])
    dataset = SmartDoc(directory=[path])
    images = []
    labels = []
    masks = []
    for img_path, norm_label in tqdm.tqdm(zip(dataset.myData[0],dataset.myData[1])):
        img = np.array(Image.open(img_path))

        # Extract the x and y coordinates
        x_cords = norm_label[[0, 2, 4, 6]]
        y_cords = norm_label[[1, 3, 5, 7]]
        x_cords = (x_cords * img.shape[1]).astype(np.int64)
        y_cords = (y_cords * img.shape[0]).astype(np.int64)
        labels.append([(i, j) for i, j in zip(x_cords, y_cords)])
        mask = gen_mask(labels[-1], sizeX=img.shape[0], sizeY=img.shape[1], sort=False)
        img = img.T
        mask = mask.T
        images.append(img)
        masks.append(mask)
    return images, masks, labels

def plot_img_and_mask(img, mask, corners, path):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[0].scatter(corners[:, 0], corners[:, 1], c='red')
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].scatter(corners[:, 0], corners[:, 1], c='red')
    else:
        ax[1].set_title(f'Ground Truth Mask')
        ax[1].imshow(mask)
        ax[1].scatter(corners[:, 0], corners[:, 1], c='red')
    plt.xticks([]), plt.yticks([])
    plt.savefig(path)
    plt.show()

if __name__ == '__main__':
    images, masks, labels = generate_dataset()
