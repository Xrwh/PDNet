
from Evison import Display, show_network
import torch
from PIL import Image
from model import convnext_base as create_model
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from torchsummary import summary
from torchvision import transforms
def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = int(np.ceil(np.sqrt(feature_map_num)))
    plt.figure()
    for index in range(1, feature_map_num + 1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index - 1], cmap='gray')
        plt.axis('off')
        # scipy.misc.imsave(str(index) + ".png", feature_map[index - 1])
        if index == feature_map_num:
            plt.savefig('/home/yang/project/ConvNeXt/visualization/' + str(index) + ".png")  # 图像保存
    plt.show()

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map)
    plt.imshow(feature_map_sum)
    plt.savefig('/home/yang/project/ConvNeXt/visualization/' + 'mto' + str(index) + ".png")  # 图像保存

if __name__ == '__main__':

    device = 'cuda:0'
    model = create_model(num_classes=2).to(device)
    PATH = '/home/yang/project/ConvNeXt/weights/best_model.pth'
    weights_dict = torch.load(PATH, map_location=device)
    model.load_state_dict(weights_dict, strict=False)

    print(show_network(model))
    print(model)
    visualized_layer = 'stages.2.0'
    img_size = (224, 224)
    shower = Display(model, visualized_layer, img_size=img_size)

    image = Image.open('/home/yang/project/ConvNeXt/data/parkinson/yuanyin5second/HC5s_Image/VA1APNITNOT56F230320170850.jpg')
    # transformer = transforms.Compose([
    #     transforms.CenterCrop(img_size)
    # ])
    # image = transformer(image)

    #特征可视化
    # feature_map =shower.generate_feature_map(image)
    # feature_visual = show_feature_map(feature_map)

    #CMA可视化
    shower.save(image)


    #model.summary()
    # summary(model, (3, 224, 224))
    # show_network(model)
