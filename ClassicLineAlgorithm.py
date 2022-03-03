import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.classic_image_processing_algorithms import e2e_algorithm
from utils.utils import gen_mask
from matplotlib import pyplot as plt
import numpy as np
def rearrange_labels(masks_pred):
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

def evaluate_net():
    # 1. Load Data loaders
    val_loader = torch.load('data/fc_data/FullyConnectedValidationDataLoader.pth')
    loss = 0
    cnt = 0
    avg_loss = []
    std_loss = []
    for batch in tqdm(val_loader):
        labels = batch['label']
        masks = batch['mask']
        # for element in range(5):
        # rand = np.random.randint(127)
        print(cnt)
        cnt += 1
        batch_loss = []
        for idx in range(masks.shape[0]):
            img = masks[idx].numpy()
            label = labels[idx].numpy()
            masks_pred = e2e_algorithm(img)
            if (masks_pred > 63).any() or (masks_pred < 0).any():
                masks_pred = e2e_algorithm(img, numPeaks=30)
            plt.figure()
            plt.imshow(img)
            plt.scatter(masks_pred[:, 0], masks_pred[:, 1], c='red')
            x = [label[idx] for idx in range(1, 8, 2)]
            y = [label[idx] for idx in range(0, 8, 2)]
            label = np.column_stack((x, y))
            plt.scatter(x, y, c='white')
            plt.savefig(f'batch_{cnt}_img_{idx}')
            masks_pred = rearrange_labels(masks_pred)
            temp_loss = np.mean(np.square(masks_pred-label))
            loss += temp_loss
            if temp_loss > 10:
                print(f'batch{cnt} - image{idx} - {temp_loss}')
                np.save(f'batch_{cnt}_img_{idx}', img)
                plt.show()
            else:
                plt.clf()
            masks_pred = rearrange_labels(masks_pred)
            temp_loss = np.mean(np.square(masks_pred-label))
            batch_loss.append(temp_loss)
        print(np.mean(batch_loss))
        print(np.std(batch_loss))
        avg_loss.append(np.mean(batch_loss))
        std_loss.append(np.std(batch_loss))
    plt.figure()
    plt.semilogy(np.arange(0, len(avg_loss)), avg_loss)
    plt.title('Loss Per Batch')
    plt.show()
    print(np.mean(avg_loss))


def dummy():
    for cnt, elem in enumerate(x):
        print(cnt)
        print(elem)

if __name__ == "__main__":
  evaluate_net()