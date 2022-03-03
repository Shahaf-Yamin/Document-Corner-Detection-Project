import cv2
import numpy as np
import evaluation
from unet import UNet
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from copy import deepcopy

def preprocess_image(img_path):
    img = Image.open(img_path)
    copy_img = deepcopy(img)
    copy_img = copy_img.resize((64, 64))
    np_img = np.array(copy_img).T
    pil_img = Image.fromarray(np_img.T.astype('uint8'), 'RGB')
    transformed_img = np.array(transforms.GaussianBlur(3)(pil_img), dtype=np.float32).T
    transformed_img = (transformed_img - 151.84097687205238) / 43.61468699572161
    return img, transformed_img


def draw_circles(img, corner, color=[0,0,256]):
    img2 = cv2.circle(img, (corner[0][0],corner[0][1]), 10, color, thickness=5)
    img2 = cv2.circle(img2, (corner[1][0],corner[1][1]), 10, color, thickness=5)
    img2 = cv2.circle(img2, (corner[2][0],corner[2][1]), 10, color, thickness=5)
    img2 = cv2.circle(img2, (corner[3][0],corner[3][1]), 10, color , thickness=5)
    return img2

    

def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagePath", default=r"data/Test/background05/tax005.avi/001.jpg", help="Path to the document image")
    parser.add_argument("-o", "--outputPath", default="Results/background05/tax005", help="Path to store the result")
    parser.add_argument("-rf", "--retainFactor", help="Floating point in range (0,1) specifying retain factor",
                        default="0.85")
    parser.add_argument("-cm", "--cornerModel", help="Model for corner point refinement",
                        default=r"checkpoints/512x512_e2e_corner_train_random_crop_near_corner_64x64_mask_concat_with_org_image_patience_2_lr_1e-05_filter_size_3_corner/Corner_Extractor_checkpoint_epoch39.pth")
    parser.add_argument("-dm", "--documentModel", help="Model for document corners detection",
                        default=r"checkpoints/64x64_e2e_model/Corner_Extractor_checkpoint_epoch3.pth")
    parser.add_argument("--load_unet", help="Model for Unet Document Segmentation",
                        default=r"checkpoints/64x64_e2e_model/Unet_checkpoint_epoch3.pth")
    parser.add_argument("--load_unet_corner", help="Model for Unet Corner Segmentation",
                        default=r"checkpoints/512x512_e2e_corner_train_random_crop_near_corner_64x64_mask_concat_with_org_image_patience_2_lr_1e-05_filter_size_3_corner/Unet_checkpoint_epoch39.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = args_processor()

    Unet = UNet(n_channels=3, n_classes=2, bilinear=True)
    Unet_corner = UNet(n_channels=3, n_classes=2, bilinear=True)
    Unet.load_state_dict(torch.load(args.load_unet, map_location='cpu'))
    Unet_corner.load_state_dict(torch.load(args.load_unet_corner, map_location='cpu'))
    Unet.eval()
    Unet_corner.eval()

    corners_extractor = evaluation.corner_extractor.GetCorners(args.documentModel, input_size=64, mode='Ours')
    corner_refiner = evaluation.corner_refiner.corner_finder(args.cornerModel, input_size=64, mode='Ours')
    for idx in range(1,50):
        print(idx)
        if idx < 10:
            img, transformed_img = preprocess_image(img_path=args.imagePath[:-5]+f'{idx}.jpg')
        else:
            img, transformed_img = preprocess_image(img_path=args.imagePath[:-6] + f'{idx}.jpg')

        with torch.no_grad():
            masks_pred = Unet(torch.tensor(transformed_img).unsqueeze(0))
            temp_masks_pred = F.one_hot(masks_pred.argmax(dim=1), Unet.n_classes).permute(0, 3, 1, 2).float()
            resized_mask = Image.fromarray(temp_masks_pred[0][1].numpy().astype('uint8')).resize(img.size)
            extracted_corners = corners_extractor.get(masks_pred, np.array(img))
            corners = corners_extractor.get_corners(masks_pred, np.array(img))
            circled_img = draw_circles(np.array(img), corners)

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

        for a in range(0, len(extracted_corners)):
            circled_img = cv2.line(circled_img, tuple(corner_address[a % 4]), tuple(corner_address[(a + 1) % 4]), (255, 0, 0), 4)
        circled_img = draw_circles(circled_img, corner_address, color=[0,256,0])
        circled_img = cv2.putText(circled_img, text='Refined Corner Estimation', org=(30, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 256, 0),
                    thickness=2, lineType=cv2.LINE_AA)
        circled_img = cv2.putText(circled_img, text='Initial Corner Estimation', org=(30, 150),
                                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 256),
                                  thickness=2, lineType=cv2.LINE_AA)
        cv2.imwrite(args.outputPath+f'/{idx}.jpg', np.array(circled_img))


