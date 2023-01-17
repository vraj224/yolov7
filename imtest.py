import math
# import matplotlib.pyplot as plt
# import torch
# import cv2
# from torchvision import transforms
# import numpy as np
# from utils.datasets import letterbox
# from utils.general import non_max_suppression_kpt
# from utils.plots import output_to_keypoint, plot_skeleton_kpts
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
# model = weigths['model']
# _ = model.float().eval()
#
# if torch.cuda.is_available():
#     model.half().to(device)
# image = cv2.imread('pressure.jpg')
# dimension = image.shape
# image = letterbox(image, 960, stride=64, auto=True)[0]
# image_ = image.copy()
# image = transforms.ToTensor()(image)
# image = torch.tensor(np.array([image.numpy()]))
#
# if torch.cuda.is_available():
#     image = image.half().to(device)
# output, _ = model(image)
# output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
# with torch.no_grad():
#     output = output_to_keypoint(output)
# nimg = image[0].permute(1, 2, 0) * 255
# nimg = nimg.cpu().numpy().astype(np.uint8)
# nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
# # image_ = letterbox(image_, dimension, stride=64, auto=True)[0]
# for idx in range(output.shape[0]):
#     plot_skeleton_kpts(image_, output[idx, 7:].T, 3)
#
# # plt.figure(figsize=(8,8))
# # plt.axis('off')
# # plt.imshow(nimg)
# # plt.imsave("multi_inf.jpg",nimg)
#
# cv2.imwrite('image_base_mod.jpg',image_)

# obj_list = [['igel_pkg 0.25', (550, 314)], ['pressure_dressing 0.78', (363, 301)]]
def obj_localize(obj_list,key_list):
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

# print(kpt_dict[1])
o_list = [['igel_pkg 0.25', (550, 314)], ['pressure_dressing 0.78', (363, 301)]]
# o_list = []
k_list = [[0, [[0, (289, 319)], [1, (397, 361)], [2, (379, 246)], [3, (494, 290)], [4, (502, 400)], [5, (584, 481)], [6, (656, 336)], [7, (738, 417)], [8, (605, 545)], [9, (720, 315)], [10, (445, 513)], [11, (615, 302)], [13, (709, 324)], [15, (759, 363)], [17, (751, 450)]]]]
# k_list = []
output = obj_localize(o_list,k_list)
print(output)
for obj in output:
    if obj:
        if len(obj) >=2:
            med_device = obj[0]
            skeleton, kpid = obj[1]
            print(f'{med_device} on {kpt_dict[kpid]} of Patient {skeleton}')