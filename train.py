#!/usr/bin/env python

import torch
import sys
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict

if __name__ == '__main__':
    '''
    This script trains a model based on the code in the dataset given
    as an argument when running the script.
    '''
   
    args = sys.argv[:]

    
    if "--save_dir" in args:
        save_dir = args[args.index("--save_dir") + 1]
    else:
        save_dir = False

    
    if "--arch" in args:
        arch = args[args.index("--arch") + 1].lower()

        if arch == 'alexnet':
            model = models.alexnet(pretrained=True)
            input_size = 9216
        elif arch == 'vgg16':
            model = models.vgg16(pretrained=True)
            input_size = 25088
        elif arch == 'densenet121':
            model = models.densenet121(pretrained=True)
            input_size = 1024
        else:
            print('Model not recongized.')
            sys.exit()
    else:
        arch = 'densenet121'
        model = models.densenet121(pretrained=True)
        input_size = 1024

    
    if "--learning_rate" in args:
        learning_rate = args[args.index("--learning_rate") + 1]
    else:
        learning_rate = 0.001

    
    if "--hidden_units" in args:
        hidden_layers = args[args.index("--hidden_units") + 1]
        hidden_layers = hidden_layers.split(',')
        hidden_layers = [int(layer) for layer in hidden_layers]
    else:
        hidden_layers = [490]

    
    output_size = 102
    hidden_layers.append(output_size)

    
    if "--epochs" in args:
        epochs = args[args.index("--epochs") + 1]
    else:
        epochs = 3

    
    if "--gpu" in args and torch.cuda.is_available():
        gpu = True
        print('\nRunning GPU...\n')
    elif "--gpu" in args and not torch.cuda.is_available():
        gpu = False
        print('\nError: Cuda not available but --gpu was set.')
        print('Running CPU...\n')
    else:
        gpu = False
        print('\nRunning CPU...\n')

    
    data_dir = str(sys.argv[1])
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"

    
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    new_data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    
    train_datasets = datasets.ImageFolder(train_dir,
                                          transform=data_transforms)

    valid_datasets = datasets.ImageFolder(valid_dir,
                                          transform=new_data_transforms)

    #
    train_loaders = DataLoader(train_datasets,
                               batch_size=32,
                               shuffle=True)

    valid_loaders = DataLoader(valid_datasets,
                               batch_size=32,
                               shuffle=True)

    class_idx = train_datasets.class_to_idx

    
    nn_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
    nn_hidden = zip(hidden_layers[:-1], hidden_layers[1:])
    nn_layers.extend([nn.Linear(x, y) for x, y in nn_hidden])

    
    for x in model.parameters():
        x.requires_grad = False

    
    params = OrderedDict()
    dropout = 0.33

    for i in range(len(nn_layers)):
        if i == 0:
            params.update({'drop{}'.format(i + 1): nn.Dropout(p=dropout)})
            params.update({'fc{}'.format(i + 1): nn_layers[i]})
        else:
            params.update({'relu{}'.format(i + 1): nn.ReLU()})
            params.update({'drop{}'.format(i + 1): nn.Dropout(p=dropout)})
            params.update({'fc{}'.format(i + 1): nn_layers[i]})

    params.update({'output': nn.LogSoftmax(dim=1)})

    
    model.classifier = nn.Sequential(params)

    
    model.cuda() if gpu else model.cpu()

    
    steps = 0
    r_loss = 0.0
    n_print = 10
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    for e in range(epochs):
        
        print('Starting training...')
        print('Training run {}...'.format((e + 1)))

        model.train()

        
        for images, labels in iter(train_loaders):
            steps += 1
            optimizer.zero_grad()

            inputs = Variable(images)
            train_labels = Variable(labels)

            
            if gpu:
                inputs = inputs.cuda()
                train_labels = train_labels.cuda()

            
            output = model.forward(inputs)
            loss = criterion(output, train_labels)
            loss.backward()
            r_loss += loss.data[0]

            
            optimizer.step()

            
            if steps % n_print == 0:
                model.eval()
                accuracy = 0.0
                valid_loss = 0.0

                for i, (images, labels) in enumerate(valid_loaders):
                    inputs = Variable(images, volatile=True)
                    valid_labels = Variable(labels, volatile=True)

                    
                    output = model.forward(inputs)
                    valid_loss += criterion(output, labels).data[0]

                    
                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Training Loss: {:.3f}.. ".format(loss))
                print("Validation Loss: {:.3f}..".format(valid_loss / len(valid_loaders)))
                print("Validation Accuracy: {:.3f}..\n".format(accuracy / len(valid_loaders)))

                r_loss = 0.0
                model.train()

    
    if save_dir:
        check_point_path = save_dir + '/checkpoint_term.pth'
    else:
        check_point_path = 'checkpoint_term.pth'

    
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'epochs': epochs,
                  'arch': arch,
                  'hidden_units': [each.out_features for each in model.classifier if
                                   hasattr(each, 'out_features') == True],
                  'learning_rate': learning_rate,
                  'class_to_idx': class_idx,
                  'optimizer_dict': optimizer.state_dict(),
                  'classifier': model.classifier,
                  'state_dict': model.state_dict()}

torch.save(checkpoint, check_point_path)
