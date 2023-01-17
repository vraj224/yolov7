import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import math
def pose_add(image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weights['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)
    # image = letterbox(image, 960, stride=64, auto=True)[0]
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

def obj_localize(obj_list,key_list):
    kpt_dict = {
        0: 'Lower Left Leg',
        1: 'Upper Left Leg',
        2: 'Lower Right Leg',
        3: 'Upper Right Leg',
        4: "Groin",
        5: 'Left Torso',
        6: 'Right Torso',
        7: 'Upper Chest',
        8: 'Upper Left Arm',
        9: 'Upper Right Arm',
        10: 'Lower Left Arm',
        11: 'Lower Right Arm',
        12: "Head",
        13: "Head",
        14: "Head",
        15: "Head",
        16: "Head",
        17: "Left Shoulder",
        18: "Right Shoulder"
    }
    final_output = []
    for obj, coord in obj_list:
        output_info = []
        min_dist = math.sqrt(2*(960**2))
        for skltn,kp_list in key_list:
            for kpid, kpcoord in kp_list:
                dist = math.hypot(kpcoord[0]-coord[0],kpcoord[1]-coord[1])
                if dist < min_dist:
                    min_dist = dist
                    output_info = [skltn,kpid]
        final_output.append([obj,output_info])


    #final output format: [obj,[skeleton #, kptid]]
    return final_output

# %matplotlib inline
# plt.figure()
# plt.axis('off')
# plt.imshow(nimg)
# plt.savefig('pressinfer.jpg')

if __name__ == '__main__':
    pass
    image_0 = cv2.imread('human.jpg')
    image_0 = letterbox(image_0, 960, stride=64, auto=True)[0]
    result, out_list = pose_add(image_0)
    print(f'outlist is {out_list}')
#     outlist format is [skeleton number, [[keypoint id1, (xcoord,ycoord)],[keypoint id2, (xcoord,ycoord)]]]
    cv2.imwrite('pressinfer2.jpg',result)