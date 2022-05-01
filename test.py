import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import gdown
from engine import train_one_epoch, evaluate
import utils
from torchvision import transforms
import label_utils
from download_dataset import download_dataset

# download dataset if it doesn't exist
if not os.path.exists('drinks'):
    download_dataset()
else:
    print("Drinks dataset already in path")

test_dict, test_classes = label_utils.build_label_dictionary("drinks/labels_test.csv")

class DrinksDataset(object):
    def __init__(self, dictionary, transform=None):
        self.dictionary = dictionary
        self.transform = transform

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        # retrieve the image filename
        key = list(self.dictionary.keys())[idx]
        # retrieve all bounding boxes
        boxes = self.dictionary[key]
        # swap xmax and ymin to conform to appropriate format
        boxes[:,1:3] = boxes[:,2:0:-1]
        # retrieve labels
        labels = boxes[:,4]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # remove label from bounding boxes
        boxes = boxes[:,0:4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        # open the file as a PIL image
        img = Image.open(key)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # apply the necessary transforms
        if self.transform:
            img = self.transform(img)
        
        # return a list of images and target (as required by the Faster R-CNN model)
        return img, target

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 4
    # use our dataset and defined transformations
    dataset_test = DrinksDataset(test_dict, transforms.ToTensor())

    # define test data loaders
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # download pretrained model weights
    if not os.path.exists('drinks_object-detection.pth'):
        url = 'https://drive.google.com/uc?id=1YlfXC1R2rH7pBb7jbxhk61XIW4QJVRnB'
        output = 'drinks_object-detection.pth'
        gdown.download(url, output, quiet=False)
        print("Downloaded pretrained model weights: drinks_object-detection.pth")
    else:
        print("Pretrained model already in path")

    # load the pretrained model weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('drinks_object-detection.pth'))
    else:
        model.load_state_dict(torch.load('drinks_object-detection.pth', map_location=torch.device('cpu')))

    # move model to the right device
    model.to(device)

    # evaluate the model on the test dataset
    evaluate(model, data_loader_test, device=device)
    
if __name__ == "__main__":
    main()
