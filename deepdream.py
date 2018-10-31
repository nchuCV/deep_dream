import numpy as np
import torch
import scipy.ndimage as nd


def objective_L2(dst, guide_features):

    return dst.data


def make_step(img, model, control=None, distance=objective_L2):

    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])

    learning_rate = 2e-2
    max_jitter = 32

    #迭代次数
    num_iterations = 1

    #结束的层数
    end_layer = 3
    guide_features = control

    for i in range(num_iterations):

        #图像随机抖动
        shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
        img = np.roll(np.roll(img, shift_x, -1), shift_y, -2)

        model.zero_grad()

        img_tensor = torch.Tensor(img)
        img_tensor = torch.tensor(img_tensor,requires_grad = True)

        act_value = model.forward(img_tensor, end_layer)

        #通过act_value.backward(diff_out),最大化特征向量act_value的 L2 范数
        #可以理解为act_value.backward(diff_out)的计算过程刚好符合L2范数的方向传播过程
        diff_out = distance(act_value, guide_features)
        act_value.backward(diff_out)

        #更新学习率
        ratio = np.abs(img_tensor.grad.data.cpu().numpy()).mean()
        learning_rate_use = learning_rate / ratio

        #因为是最大化特征(目标),所以更新参数时是 +
        img_tensor.data.add_(img_tensor.grad.data * learning_rate_use)

        # b, c, h, w
        img = img_tensor.data.cpu().numpy()

        #逆抖动
        img = np.roll(np.roll(img, -shift_x, -1), -shift_y, -2)

        img[0, :, :, :] = np.clip(img[0, :, :, :], -mean / std,(1 - mean) / std)

    return img


def dream(model,base_img,octave_n=6,octave_scale=1.4,control=None,distance=objective_L2):

    octaves = [base_img]

    #得到图像金字塔
    for i in range(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale),order=1))


    detail = np.zeros_like(octaves[-1])

    #octaves[::-1]将octaves里的元素顺序反转
    for octave, octave_base in enumerate(octaves[::-1]):

        #h,w图片高，宽
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        input_oct = octave_base + detail
        print(input_oct.shape)
        out = make_step(input_oct, model, control, distance=distance)
        detail = out - octave_base

    return out


