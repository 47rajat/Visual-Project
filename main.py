'''
model84: batch_size=30, num_epoch=5, learning_rate=0.01,resnet18,kmax=3,kmin=3
model93: batch_size=30, num_epoch=5, learning_rate=0.1,resnet18,kmax=3,kmin=3
'''
import resnet_weldon
import dataset_loader
import torchvision.transforms as transforms
from torch.autograd import Variable
import datetime
import torch
import numpy as np

print('version 3.7')
torch.cuda.device(1)
print(torch.cuda.device_count())

# Using the VOC2007 dataset provided for assignment 2
classes = ('car','bicycle',
           'aeroplane', 'cat', 'dog', 'bird', 'boat',
           'bottle', 'bus', 'chair',
           'cow', 'diningtable', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

num_classes = len(classes)

# defining model
model = resnet_weldon.resnet18_weldon(num_classes=num_classes, pretrained=True, kmax=3,kmin=3)

# parameters
resnet_input = 224
batch_size = 100
num_epochs = 3
learning_rate = 0.1
threshold = -5.5

print('Number of epochs = {}, batch_size = {}, learning rate = {}, threshold = {}'.format(num_epochs, batch_size, learning_rate, threshold))

# tranformation
composed_transform = transforms.Compose([transforms.Scale((resnet_input,resnet_input)),
                                         transforms.ToTensor()])

criterion = torch.nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_weldon():
    start_time = datetime.datetime.now()
    train_loader, len_dataset = dataset_loader.create_data_loader(batch_size=batch_size,train=True, transform=composed_transform)
    print(len_dataset)
    for epoch in range(num_epochs):
        ep_loss = 0

        counter = 0
        for image, label in train_loader:
            # print(image.size())
            # print('Time passed: {}'.format(datetime.datetime.now() - start_time))
            counter += 1
            # print(label)
            image = Variable(image)
            label = Variable(label)
            
            optimizer.zero_grad()
            output = model.forward(image.view(-1, 3, resnet_input, resnet_input))
            # print(output)
            loss = criterion(output, label)
            # print(loss)
            # assert(False)
            loss.backward()
            optimizer.step()

            ep_loss += loss.data[0]

            if counter % 25  == 0:
                print('Time passed: {}'.format(datetime.datetime.now() - start_time))
                print('Epoch {}/{}, step: {}, loss of this step = {}'.format(epoch+1, num_epochs, counter, loss.data[0]/output.size()[0]))
                print()

        print('Average loss for epoch {} is {}'.format(epoch + 1, ep_loss/len_dataset))

        torch.save(model.state_dict(), 'model.pkl')


def test_weldon(model):
    start_time = datetime.datetime.now()
    test_loader, len_dataset = dataset_loader.create_data_loader(batch_size=batch_size,train=False, transform=composed_transform)
    print(len_dataset)

    step = 0
    for image, label in test_loader:
        image = Variable(image)

        output = model.forward(image)

        # smax = torch.nn.Softmax()
        # output = smax(output)
        output = output.data.numpy()
        label = label.numpy()

        step += output.shape[0]
        # print(output)
        # print(label)
        # assert(False)

        # for each image in batch size
        for b in range(output.shape[0]):
            # for each prediction made for an image
            for idx in range(len(classes)):
                # if prediction > 0, class is present.
                pred = output[b][idx]
                if pred >= threshold:
                    # stroing the score for that prediction
                    predictions[classes[idx]][0].append(pred)

                    # checking if class present in image
                    if label[b][idx] == 1.0:
                        predictions[classes[idx]][1].append(1)
                    else:
                        predictions[classes[idx]][1].append(0)
                
                # storing all occurence of all classes
                if label[b][idx] == 1.0:
                    predictions[classes[idx]][2] += 1 
        
        if step % 100 == 0:
            print('Time passed: {}'.format(datetime.datetime.now() - start_time))
            print('{} Images done'.format(step))
            print()

    
def compute_map(predictions):
    maP_score = []
    for c in classes:
        aP_score = 0.0
        scores = predictions[c][0]
        output = predictions[c][1]
        total_count = predictions[c][2]

        print('Prediction for class {} is {} and total is {}'.format(c,sum(output),total_count))

        scores = np.array(scores)

        idx = np.argsort(-1*scores)

        correct = 0
        counter = 0
        for i in idx:
            counter += 1
            if output[i] == 1:
                correct += 1
                aP_score += (correct/counter)
        
        other_ap = aP_score/total_count
        
        if correct != 0:
            aP_score /= correct
        else:
            aP_score = 0

        maP_score.append((aP_score,other_ap))
    
    counter = 0
    for c in classes:
        print('Average Precision for class {} is {}'.format(c,maP_score[counter]))
        counter += 1

    ans = 0
    ans2 = 0
    for score in maP_score:
        ans += score[0]
        ans2 += score[1]
    # ans = sum(maP_score)/len(maP_score)
    ans /= len(maP_score)
    ans2 /= len(maP_score)

    print('Final mAP score is {} and new one is  {}'.format(ans, ans2))







if __name__ == '__main__':
    # train_weldon()

    predictions = {}

    for c in classes:
        predictions[c] = [[], [],0]
    
    model = resnet_weldon.resnet18_weldon(num_classes=num_classes, pretrained=True, kmax=3, kmin=3)
    model.load_state_dict(torch.load('model.pkl')) 

    test_weldon(model)

    compute_map(predictions)
    


