# Transformer-based Keyword Spotting
David Angelo Piscasio's implementation of a transformer-based Keyword Spotting (KWS) application for UP Diliman EEEI's EE197-Z course.

### Model reference
The following implementation makes use of a transformer model to perform keyword spotting. To know more about the transformer model reference, you may check out the following link to the code:
* [Transformer for CIFAR10]([https://arxiv.org/abs/1506.01497](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/transformer/python/transformer_demo.ipynb))

### Install requirements
To install the needed dependencies for this implementation,
```
pip install -r requirements.txt
```

### Training the model
To train the model on the train dataset and test its performance on the test dataset,
```
python3 train.py
```

### KWS Demonstration
To try out the KWS demo on a GUI-based application,
```
python3 kws-infer.py
```
