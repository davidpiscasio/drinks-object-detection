# Object Detection with Drinks Dataset using Faster R-CNN
David Angelo Piscasio's implementation of object detection with the drinks dataset as a requirement for UP Diliman EEEI's EE197-Z course.

### Model reference
The following implementation makes use of Faster R-CNN to perform object detection. To know more about the model, you may check out the links to the paper and code:
* [Paper](https://arxiv.org/abs/1506.01497)
* [Code](https://github.com/rbgirshick/py-faster-rcnn)

### Data preparation
This object detection model makes use of the drinks dataset. You may download the dataset together with the labels from [https://bit.ly/adl2-ssd](https://bit.ly/adl2-ssd). However, you do not need to download this if you will be running ```train.py``` or ```test.py``` since the scripts will automatically prepare the data for you if the dataset is still not present in the path.

### Install requirements
To install the needed dependencies for this implementation,
```
pip install -r requirements.txt
```

### Testing the model
To test the model on the test dataset with the pre-trained model weights from ```drinks_object-detection.pth```,
```
python3 test.py
```
The script will automatically download the pre-trained model weights and load it to the model.

### Training the model
To train the model on the train dataset and test its performance on the test dataset,
```
python3 train.py
```
The script will also automatically save the newly trained model weights to the filename ```drinks_object-detection.pth```.

### Other references
Here are other references that were used to implement the object detection program:
* [Torchvision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
* [PyTorch Vision Detection Reference](https://github.com/pytorch/vision/tree/main/references/detection)
