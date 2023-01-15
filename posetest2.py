import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

def pose_add(image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weights['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)
        # print('if here')
    output, _ = model(image)

    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)
    # print(output)
    # nimg = image[0].permute(1, 2, 0) * 255
    # nimg = nimg.cpu().numpy().astype(np.uint8)
    # nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    key_list = []
    for idx in range(output.shape[0]):
        # print(output[idx, 7:].T)
        id_list = plot_skeleton_kpts(image_, output[idx, 7:].T, 3)
        # print(f'id list is {id_list}')
        key_list.append([idx,id_list])
    return [image_, key_list]

# %matplotlib inline
# plt.figure()
# plt.axis('off')
# plt.imshow(nimg)
# plt.savefig('pressinfer.jpg')

if __name__ == '__main__':
    pass
#     image_0 = cv2.imread('multi_2.jpg')
#     result, out_list = pose_add(image_0)
#     print(f'outlist is {out_list}')
# #     outlist is [skeleton number, [[keypoint id1, (xcoord,ycoord)],[keypoint id2, (xcoord,ycoord)]]]
# cv2.imwrite('pressinfer.jpg',result)