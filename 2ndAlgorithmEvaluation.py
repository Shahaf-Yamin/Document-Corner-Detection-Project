import numpy as np
import evaluation
import os
import xml.etree.ElementTree as ET
import utils.utils as utils
from matplotlib.path import Path
import torch
from unet import UNet
from torch import Tensor
from PIL import Image
from copy import deepcopy
from torchvision import transforms
from matplotlib import pyplot as plt

def preprocess_image(img_path, mode):
    img = Image.open(img_path)
    copy_img = deepcopy(img)
    if mode == 'Ours':
        copy_img = copy_img.resize((64, 64))
        np_img = np.array(copy_img).T
        pil_img = Image.fromarray(np_img.T.astype('uint8'), 'RGB')
        transformed_img = np.array(transforms.GaussianBlur(3)(pil_img), dtype=np.float32).T
        transformed_img = (transformed_img - 151.84097687205238) / 43.61468699572161
    else:
        transformed_img = copy_img.resize((32, 32))
        np_img = np.array(transformed_img).T
        pil_img = Image.fromarray(np_img.T.astype('uint8'), 'RGB')
        transformed_img = np.array(transforms.ToTensor()(pil_img), dtype=np.float32).T
    return img, transformed_img

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
    return mask

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagePath", default=r"data/Test/background01/tax005.avi/001.jpg", help="Path to the document image")
    parser.add_argument("-o", "--outputPath", default="Results/background01/tax005", help="Path to store the result")
    parser.add_argument("-rf", "--retainFactor", help="Floating point in range (0,1) specifying retain factor",
                        default="0.85")
    parser.add_argument("-cm", "--cornerModel", help="Model for corner point refinement",
                        default=r"checkpoints/512x512_e2e_corner_train_random_crop_near_corner_64x64_mask_concat_with_org_image_patience_2_lr_1e-05_filter_size_3_corner/Corner_Extractor_checkpoint_epoch39.pth")
    parser.add_argument("-dm", "--documentModel", help="Model for document corners detection",
                        default=r"checkpoints/64x64_e2e_model/Corner_Extractor_checkpoint_epoch3.pth")
    # parser.add_argument("-cm", "--cornerModel", help="Model for corner point refinement",
    #                     default=r"checkpoints/bench_mark_corner_trainlr_0.005_corner/Corner_Extractor_checkpoint_epoch49.pth")
    # parser.add_argument("-dm", "--documentModel", help="Model for document corners detection",
    #                     default=r"checkpoints/bench_mark_document_trainlr_0.005_document/Corner_Extractor_checkpoint_epoch49.pth")
    parser.add_argument("--load_unet", help="Model for Unet Document Segmentation",
                        default=r"checkpoints/64x64_e2e_model/Unet_checkpoint_epoch3.pth")
    parser.add_argument("--load_unet_corner", help="Model for Unet Corner Segmentation",
                        default=r"checkpoints/512x512_e2e_corner_train_random_crop_near_corner_64x64_mask_concat_with_org_image_patience_2_lr_1e-05_filter_size_3_corner/Unet_checkpoint_epoch39.pth")
    parser.add_argument("--mode", help="Which Model should we use to evaluate preformance",
                        default=r"Ours", choices=['benchMark', 'Ours'])
    return parser.parse_args()

if __name__ == "__main__":
    args = args_processor()
    output_path = "../output.jpg"
    Unet = UNet(n_channels=3, n_classes=2, bilinear=True)
    Unet_corner = UNet(n_channels=3, n_classes=2, bilinear=True)
    Unet.load_state_dict(torch.load(args.load_unet, map_location='cpu'))
    Unet_corner.load_state_dict(torch.load(args.load_unet_corner, map_location='cpu'))
    MODE = args.mode
    Unet.eval()
    Unet_corner.eval()
    corners_extractor = evaluation.corner_extractor.GetCorners(args.documentModel, input_size=32 if MODE == 'benchMark' else 64, mode=MODE)
    corner_refiner = evaluation.corner_refiner.corner_finder(args.cornerModel, input_size=32 if MODE == 'benchMark' else 64, mode=MODE)

    bg_path = r"data/Test"
    DSC_sum = np.zeros((5,))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corners_extractor.model.to(device)
    corner_refiner.model.to(device)
    Unet_corner.to(device)
    Unet.to(device)
    Results = {}
    for bg_idx, bg in enumerate([f for f in os.listdir(bg_path) if os.path.isdir(os.path.join(bg_path, f))]):
        imgs_counter = 0
        bg_Results = []
        print(bg)
        path = os.path.join(bg_path, bg)
        for video in [g for g in os.listdir(path) if os.path.isdir(os.path.join(path, g))]:
            imgs_path = os.path.join(path, video)
            list_gt = []
            tree = ET.parse(imgs_path + "/" + video + ".gt")
            root = tree.getroot()
            for a in root.iter("frame"):
                list_gt.append(a)

            for img in [e for e in os.listdir(imgs_path) if os.path.join(imgs_path, e).endswith('.jpg')]:
                open_img, transformed_img = preprocess_image(img_path=os.path.join(imgs_path,img), mode=MODE)
                if MODE == 'Ours':
                    masks_pred = Unet(torch.tensor(transformed_img).unsqueeze(0).to(device))
                else:
                    masks_pred = torch.tensor(transformed_img.T).unsqueeze(0)
                extracted_corners = corners_extractor.get(masks_pred, np.array(open_img))
                corners = corners_extractor.get_corners(masks_pred, np.array(open_img))
                
                list_of_points = {}
                for point in list_gt[int(float(img[0:-4])) - 1].iter("point"):
                    myDict = point.attrib

                    list_of_points[myDict["name"]] = (
                        int(float(myDict['x'])), int(float(myDict['y'])))

                ground_truth = np.asarray(
                    (list_of_points["tl"], list_of_points["tr"], list_of_points["br"],
                     list_of_points["bl"]))
                ground_truth = utils.sort_gt(ground_truth)
                # circled_img = draw_circles(curr_img, ground_truth)

                corner_address = []
                # Refine the detected corners using corner refiner
                image_name = 0
                for corner in extracted_corners:
                    image_name += 1
                    corner_img = corner[0]
                    refined_corner = np.array(corner_refiner.get_location(corner_img, Unet_corner, 0.85))
            
                    # Converting from local co-ordinate to global co-ordinates of the image
                    refined_corner[0] += corner[1]
                    refined_corner[1] += corner[2]
            
                    # Final results
                    corner_address.append(refined_corner)
            
            
                gt_mask = gen_mask(list(ground_truth), 1920, 1080, False)
                mask = gen_mask(list(corner_address), 1920, 1080, False)
                dsc = dice_coeff(Tensor(mask), Tensor(gt_mask)).detach().numpy()
                bg_Results.append(dsc)
                imgs_counter += 1
                DSC_sum[bg_idx] += (dsc - DSC_sum[bg_idx]) / imgs_counter
                print(DSC_sum[bg_idx])

        Results[int(bg[-1])] = np.array(bg_Results)
    print(DSC_sum)
    plt.figure()
    res = []
    for idx in range(1,6):
        res.append(Results[idx])
    plt.boxplot(res, positions=[2, 4, 6, 8, 10], showfliers=False)
    plt.grid()
    plt.xticks([2, 4, 6, 8, 10], ['BG 1', 'BG 2', 'BG 3', 'BG 4', 'BG 5'], rotation=20)
    plt.savefig(f'{MODE}_DiceBoxPlot', dpi=512)

