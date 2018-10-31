import torch
from torchvision import transforms
from PIL import Image
from resnet import resnet50
from deepdream import dream
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
from util import img_cvt
import os


def main(file_name):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_img = Image.open(file_name)
    input_tensor = img_transform(input_img).unsqueeze(0)
    input_np = input_tensor.numpy()

    # load model
    model = resnet50(pretrained=True)
    if torch.cuda.is_available():
        model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False

    if os.path.exists('outtmp.pkl'):
        fr = open('outtmp.pkl', 'rb')
        out = pickle.load(fr)
        out = img_cvt(out)
        out = np.array(out, dtype=np.uint8)
        save_out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite('deep_dream.jpg', save_out)
    else:
        out = dream(model, input_np)
        fw = open('outtmp.pkl','wb')
        pickle.dump(out,fw)
        out = img_cvt(out)
        out = np.array(out, dtype=np.uint8)
        save_out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite('deep_dream.jpg', save_out)

    plt.imshow(out)
    plt.show()

if __name__ == '__main__':

    file_name = 'sky.jpg'
    main(file_name)