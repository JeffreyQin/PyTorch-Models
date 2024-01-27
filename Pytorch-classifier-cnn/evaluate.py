from cnn import CNN
from data import classes, test_dataloader
import torch
from tqdm import tqdm

PATH = './Pytorch-classifier-CNN/cifar_net.pth'
model = CNN()
model.load_state_dict(torch.load(PATH))

""" MINOR PREDICTION TEST """

images, labels = next(iter(test_dataloader))
outputs = model(images)

_, predicted = torch.max(outputs, 1) # 1 is the dimension on which max is calculated
# torch.max(outputs, 1) returns two tensors of batch_size dimensions
# entry i in first tensor is the value of maximum predicted class in sample i in the class
# entry i in second tensor is the index of maximum predicted class in sample i in the class

print('sample prediction test result: predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

""" MAJOR PREDICTION TEST """

correct = 0
total = 0

model.eval()

with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        
        prediction = model(images)
        _, predicted = torch.max(prediction.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # (predicted == labels) creates a tensor of batch_size dimensions, each is True or False

print(f'overall accuracy is {100 * correct // total} %')

""" PREDICTION TEST TO CHECK ACCURACY FOR EACH CLASS """

correct_each_class = {classname: 0 for classname in classes}
total_each_class = {classname: 0 for classname in classes}

with torch.no_grad():
    for (images, labels) in test_dataloader:

        prediction = model(images)
        _, predicted = torch.max(prediction, 1)

        for single_predicted, label in zip(predicted, labels):
            total_each_class[classes[label]] += 1
            if label == single_predicted:
                correct_each_class[classes[label]] += 1

for classname, correct_count in correct_each_class.items():
    print(f'accuracy for class {classname:5s} is {100 * float(correct_count) / total_each_class[classname]}%')
