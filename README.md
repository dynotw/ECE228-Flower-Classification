# ECE228 - Flower Classification by SVM and CNN

## Group
* Goup5:
  * Junfeng Zhao
  * Linyan Zheng
  * Yening Dong

## CNN Code
### Firstly you need to install neccessary python packages:
1. numpy
2. pandas
3. time
4. globNum
5. torch
6. torchvision
7. matplotlib

For installing these packages, you can use either pip to install packages. For example,
```
pip install numpy
```

### Preview Flower Dataset
This part code is for users to have a brief picture of the dataset

### Data Process
In this part, users can change transformation:
For installing these packages, you can use either pip to install packages. For example,
```
train_transforms = transforms.Compose([
                                        transforms.RandomChoice([
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.RandomRotation(30),
                                        ]),
                                        
                                        transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
```
You can use different transforms.xxx functions to achieve different data transformations

### Load Model
You load different model in this code, please choose pretrained = True
```
model1 = models.vgg16(pretrained=True)
model1
```

### Define Classier & use GPU
You should define a classifer, depending on input and output.
```
model1.classifier = nn.Sequential(nn.Linear(512*7*7, 4096),
                                  nn.ReLU(True),
                                  nn.Dropout(),
                                  nn.Linear(4096, 1024),
                                  nn.ReLU(True),
                                  nn.Dropout(),
                                  nn.Linear(1024, 102))
```
You can choose different loss functions:
```
criterion = nn.CrossEntropyLoss() # defining loss function
```

### The following
The left part just train, visuialize and store results
