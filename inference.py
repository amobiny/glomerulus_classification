import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.dataset import Data as data
from utils.visualize_utils import visualize
from utils.eval_utils import compute_accuracy
from models import *
import numpy as np
from config import options

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


@torch.no_grad()
def evaluate():
    net.eval()
    test_loss = 0
    targets, outputs = [], []

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = net(data)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            test_loss += batch_loss

        test_loss /= (batch_id + 1)
        test_acc = compute_accuracy(torch.cat(targets), torch.cat(outputs))

        # display
        log_string("validation_loss: {0:.4f}, validation_accuracy: {1:.02%}"
                   .format(test_loss, test_acc))


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.dirname(os.path.dirname(options.load_model_path))

    LOG_FOUT = open(os.path.join(save_dir, 'log_inference.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    # bkp of inference
    os.system('cp {}/inference.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    if options.model == 'resnet':
        net = resnet.resnet50()
        net.fc = nn.Linear(net.fc.in_features, options.num_classes)
        grad_cam_hooks = {'forward': net.layer4, 'backward': net.fc}
    elif options.model == 'vgg':
        net = vgg19_bn(pretrained=True, num_classes=options.num_classes)
        grad_cam_hooks = {'forward': net.features, 'backward': net.fc}
    elif options.model == 'inception':
        net = inception_v3(pretrained=True)
        net.aux_logits = False
        net.fc = nn.Linear(2048, options.num_classes)
        grad_cam_hooks = {'forward': net.Mixed_7c, 'backward': net.fc}
    elif options.model == 'densenet':
        net = densenet.densenet121()
        net.classifier = nn.Linear(net.classifier.in_features, out_features=options.num_classes)
        grad_cam_hooks = {'forward': net.features.norm5, 'backward': net.classifier}

    log_string('{} model Generated.'.format(options.model))
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in net.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.cuda()
    net = nn.DataParallel(net)

    ##################################
    # Load the trained model
    ##################################
    ckpt = options.load_model_path
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    log_string('Model successfully loaded from {}'.format(ckpt))

    ##################################
    # Loss and Optimizer
    ##################################
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=options.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################
    os.system('cp {}/dataset/dataset.py {}'.format(BASE_DIR, save_dir))

    train_dataset = data(mode='train', data_len=options.data_len)
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)
    test_dataset = data(mode='test', data_len=options.data_len)
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start Testing')
    evaluate()

    #################################
    # Grad Cam visualizer
    #################################
    if options.gradcam:
        log_string('Generating Gradcam visualizations')
        iter_num = options.load_model_path.split('/')[-1].split('.')[0]
        img_dir = os.path.join(save_dir, 'imgs')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        viz_dir = os.path.join(img_dir, iter_num)
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        visualize(net, test_loader, grad_cam_hooks, viz_dir)
        log_string('Images saved in: {}'.format(viz_dir))




