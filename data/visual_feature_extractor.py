import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile
import torch
import os


ImageFile.LOAD_TRUNCATED_IMAGES = True


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的最后层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.resnet_layer(x)
        return x


# resnet50 = models.resnet50(pretrained=True, progress=True)
resnet18 = models.resnet18(pretrained=True, progress=True)
model = Net(resnet18)
print(model)  # output size 16*512*1*1


class img_Dataset(Dataset):
    def __init__(self, root, resize):
        self.image_files = np.array([x.path for x in os.scandir(root) if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")])
        self.transform = transforms.Compose([transforms.Resize(size=(resize, resize))])
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, index):
        path = self.image_files[index]
        # print(path)
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            img = self.toTensor(img)
            # print(img.shape)
        except:
            img = np.zeros((3, 224, 224))
            img = torch.Tensor(img)
        return img

    def __len__(self):
        return len(self.image_files)


file_paths = ["/data/FakeNewsNet/PolitiFact/Fake_News/img/", "/data/FakeNewsNet/PolitiFact/Real_News/img/", "/data/FakeNewsNet/GossipCop/Fake_News/img/", "/data/FakeNewsNet/GossipCop/Real_News/img/"]
for file_path in file_paths:
    all_img = os.listdir(file_path)
    print(all_img)
    dataset = img_Dataset(file_path, 224)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    EPOCH = 1


    for epoch in range(EPOCH):
        for step, data in enumerate(train_loader):
            txt_name = all_img[step][:-4]
            out = model(data)
            out = torch.reshape(out, (out.shape[1], -1))
            print(step)
            out_np = out.detach().numpy()
            np.savetxt(file_path[:-4] + 'visual_feature/' + txt_name + '.txt', out_np)


    empty = np.zeros((1,3,224,224))
    empty = torch.Tensor(empty)
    out = model(empty)
    out = torch.reshape(out, (out.shape[1], -1))
    out_np = out.detach().numpy()
    np.savetxt(file_path[:-4] + 'visual_feature/' + 'white.txt', out_np)
