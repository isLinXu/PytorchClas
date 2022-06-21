"""
simple classifier for both train and predict
on single image

the classify image data is using flowers


At least 40 epochs you can see converge evidence
Epoch: 43  iter: 100, loss: 0.37397363781929016
Epoch: 43  iter: 200, loss: 0.7303532958030701
Epoch: 43  iter: 300, loss: 0.8449480533599854
Epoch: 44  iter: 100, loss: 0.5675420165061951
Epoch: 44  iter: 200, loss: 0.8853188157081604
Epoch: 44  iter: 300, loss: 0.9220790863037109
Checking prediction ouput...
output vs target:
[0 2 1 0 4 2 3 1 4 3]
[0 2 0 4 4 4 4 1 4 3]
accuracy: 60.0%


"""
from classifier_trainer import Trainer
from nets.mobilenet_v2 import MobileNetV2
import sys
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
from alfred.dl.torch.common import device


target_size = 224
num_classes = 5


def test_data():
    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )
    # write some data loader
    dataset = ImageFolder('./data/flower_photos/', transform=transform,)
    print(dataset.class_to_idx)
    dataloader = DataLoader(dataset=dataset, batch_size=24, shuffle=True, num_workers=1)
    for i, batch_data in enumerate(dataloader):
        if i == 0:
            img, target = batch_data
            img = img.numpy()
            target = target.numpy()

            print(img)
            print(target)
            print(img.shape)
            print(target.shape)
            cv2.imshow('rr', np.transpose(img[0], [1, 2, 0]))
            # cv2.imshow('rrppp', np.transpose(target[0], [1, 2, 0]))
            cv2.waitKey(0)


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )

    # write some data loader
    dataset = ImageFolder('./data/flower_photos/', transform=transform)
    print(dataset.class_to_idx)
    num_classes = len(dataset.classes)
    dataloader = DataLoader(dataset=dataset, batch_size=12, shuffle=True, num_workers=0)

    model = MobileNetV2(num_classes=num_classes, input_size=target_size)
    trainer = Trainer(model=model, train_loader=dataloader, val_loader=None, save_epochs=50,
                      checkpoint_dir='./checkpoints', resume_from='checkpoint.pth.tar', 
                      num_epochs=100)
    trainer.train()


def predict(img_f):
    img = cv2.imread(img_f)
    model = MobileNetV2(num_classes=num_classes, input_size=target_size)
    model_path = './checkpoints/checkpoint.pth.tar'
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()

    print('model loaded, ready to predict.')
    in_img = cv2.resize(img, (target_size, target_size))
    in_img = torch.tensor(in_img).unsqueeze().to(device)
    print('in img shape: {}'.format(in_img.szie()))
    pred = model(in_img)
    print(pred)
    print('the largest index is your classes, avaliable class are: {}'.format())


if __name__ == '__main__':
    if len(sys.argv) >= 2:

        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'predict':
            img_f = sys.argv[2]
            predict(img_f)
        elif sys.argv[1] == 'preview':
            test_data()
    else:
        print('python3 classifier.py train to train net'
              '\npython3 classifier.py predict img_f/path to predict img.')

