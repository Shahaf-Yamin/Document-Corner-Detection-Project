
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import modelfactory


class GetCorners:
    def __init__(self, checkpoint_dir, input_size, mode):
        self.model = modelfactory.MaskedDocumentResnet(num_classes=8, numInputChannels=2 if mode == 'Ours' else 3, input_size=input_size)
        self.model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def get(self, mask, image_array):
        with torch.no_grad():
            # mask_temp = mask.unsqueeze(0)
            if torch.cuda.is_available():
                mask = mask.cuda()

            model_prediction = self.model(mask).cpu().data.numpy()[0]

            model_prediction = np.array(model_prediction)

            x_cords = model_prediction[[0, 2, 4, 6]]
            y_cords = model_prediction[[1, 3, 5, 7]]

            x_cords = x_cords * image_array.shape[1]
            y_cords = y_cords * image_array.shape[0]

            # Extract the four corners of the image. Read "Region Extractor" in Section III of the paper for an explanation.

            top_left = image_array[
                       max(0, int(2 * y_cords[0] - (y_cords[3] + y_cords[0]) / 2)):int((y_cords[3] + y_cords[0]) / 2),
                       max(0, int(2 * x_cords[0] - (x_cords[1] + x_cords[0]) / 2)):int((x_cords[1] + x_cords[0]) / 2)]

            top_right = image_array[
                        max(0, int(2 * y_cords[1] - (y_cords[1] + y_cords[2]) / 2)):int((y_cords[1] + y_cords[2]) / 2),
                        int((x_cords[1] + x_cords[0]) / 2):min(image_array.shape[1] - 1,
                                                               int(x_cords[1] + (x_cords[1] - x_cords[0]) / 2))]

            bottom_right = image_array[int((y_cords[1] + y_cords[2]) / 2):min(image_array.shape[0] - 1, int(
                y_cords[2] + (y_cords[2] - y_cords[1]) / 2)),
                           int((x_cords[2] + x_cords[3]) / 2):min(image_array.shape[1] - 1,
                                                                  int(x_cords[2] + (x_cords[2] - x_cords[3]) / 2))]

            bottom_left = image_array[int((y_cords[0] + y_cords[3]) / 2):min(image_array.shape[0] - 1, int(
                y_cords[3] + (y_cords[3] - y_cords[0]) / 2)),
                          max(0, int(2 * x_cords[3] - (x_cords[2] + x_cords[3]) / 2)):int(
                              (x_cords[3] + x_cords[2]) / 2)]

            top_left = (top_left, max(0, int(2 * x_cords[0] - (x_cords[1] + x_cords[0]) / 2)),
                        max(0, int(2 * y_cords[0] - (y_cords[3] + y_cords[0]) / 2)))
            top_right = (
            top_right, int((x_cords[1] + x_cords[0]) / 2), max(0, int(2 * y_cords[1] - (y_cords[1] + y_cords[2]) / 2)))
            bottom_right = (bottom_right, int((x_cords[2] + x_cords[3]) / 2), int((y_cords[1] + y_cords[2]) / 2))
            bottom_left = (bottom_left, max(0, int(2 * x_cords[3] - (x_cords[2] + x_cords[3]) / 2)),
                           int((y_cords[0] + y_cords[3]) / 2))

            return top_left, top_right, bottom_right, bottom_left
    def get_corners(self, mask, image):
        with torch.no_grad():

            if torch.cuda.is_available():
                mask = mask.cuda()

            model_prediction = self.model(mask.float()).cpu().data.numpy()[0]

            model_prediction = np.array(model_prediction)

            x_cords = model_prediction[[0, 2, 4, 6]]
            y_cords = model_prediction[[1, 3, 5, 7]]

            x_cords = x_cords * image.shape[1]
            y_cords = y_cords * image.shape[0]
        return  [(i, j) for i,j in zip(x_cords, y_cords)]