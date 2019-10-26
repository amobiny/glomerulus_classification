import os
import warnings
import torch.backends.cudnn as cudnn

warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from models import *

from torch.optim import Adam
import numpy as np
from config import options
import torch
import torch.nn.functional as F
from utils.eval_utils import compute_accuracy
import torch.nn as nn
import torch
from dataset.dataset import Data as data

# Temporarily writing to file not rendering to window.
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


@torch.no_grad()
def evaluate():
    net.eval()
    test_loss = 0
    targets, predictions = [], []

    for batch_id, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        target_ohe = F.one_hot(target, options.num_classes)
        y_pred = F.log_softmax(net(data), dim=1)
        batch_loss = F.nll_loss(y_pred, target)

        targets += [target_ohe]
        predictions += [y_pred]
        test_loss += batch_loss

    test_loss /= (len(test_loader) * options.batch_size)
    test_acc = compute_accuracy(torch.cat(targets), torch.cat(predictions))

    # display
    log_string("validation_loss: {0:.4f}, validation_accuracy: {1:.02%}"
               .format(test_loss, test_acc))


@torch.enable_grad()
def grad_cam(model, x, hooks, cls_idx=None):
    """ cf CheXpert: Test Results / Visualization; visualize final conv layer, using grads of final linear layer as weights,
    and performing a weighted sum of the final feature maps using those weights.
    cf Grad-CAM https://arxiv.org/pdf/1610.02391.pdf """
    model.eval()
    model.zero_grad()

    # register backward hooks
    conv_features, linear_grad = [], []
    forward_handle = hooks['forward'].register_forward_hook(
        lambda module, in_tensor, out_tensor: conv_features.append(out_tensor))
    backward_handle = hooks['backward'].register_backward_hook(
        lambda module, grad_input, grad_output: linear_grad.append(grad_input))

    # run model forward and create a one hot output for the given cls_idx or max class
    outputs = model(x)
    if not cls_idx: cls_idx = outputs.argmax(1)
    one_hot = F.one_hot(cls_idx, outputs.shape[1]).float().requires_grad_(True)

    # run model backward
    one_hot.mul(outputs).sum().backward()

    # compute weights; cf. Grad-CAM eq 1 -- gradients flowing back are global-avg-pooled to obtain the neuron importance weights
    weights = linear_grad[0][2].mean(1).view(1, -1, 1, 1)
    # compute weighted combination of forward activation maps; cf Grad-CAM eq 2; linear combination over channels
    cam = F.relu(torch.sum(weights * conv_features[0], dim=1, keepdim=True))

    # normalize each image in the minibatch to [0,1] and upscale to input image size
    cam = cam.clone()  # avoid modifying tensor in-place

    def norm_ip(t, min, max):
        t.clamp_(min=min, max=max)
        t.add_(-min).div_(max - min + 1e-5)

    for t in cam:  # loop over mini-batch dim
        norm_ip(t, float(t.min()), float(t.max()))

    cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=True)

    # cleanup
    forward_handle.remove()
    backward_handle.remove()
    model.zero_grad()

    return cam


def visualize(model, dataloader, grad_cam_hooks):
    # 1. run through model to compute logits and grad-cam
    save_dir = '/home/cougarnet.uh.edu/mpadmana/Downloads/ResNet'
    imgs, labels, scores, masks, idxs = [], [], [], [], []
    for batch_idx, (x, target) in enumerate(dataloader):
        imgs += [x]
        labels += [target]
        # idxs += idx.tolist()
        # x = x.to(args.device)
        scores += [model(x).cpu()]
        # scores += [model(x).cpu()]
        masks += [grad_cam(model, x, grad_cam_hooks).cpu()]
        imgs, labels, scores, masks = torch.cat(imgs), torch.cat(labels), torch.cat(scores), torch.cat(masks)

        for img_id, (img, mask) in enumerate(zip(imgs, masks)):
            fig, ax = plt.subplots(1, 2)
            ax[0].set_title('Original image', fontsize=10)
            img = img.view(img.shape[1], img.shape[2], img.shape[0])
            ax[0].imshow(img, cmap='gray')
            ax[1].set_title('Top class activation')
            ax[1].imshow(img.squeeze(), cmap='gray')
            ax[1].imshow(mask.squeeze(), cmap='jet', alpha=0.5)
            print(1)
            plt.savefig(os.path.join(save_dir, '{}-{}.png'.format(batch_idx, img_id)), dpi=300, bbox_inches='tight')


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
        net = resnet.resnet34(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, options.num_classes)
        grad_cam_hooks = {'forward': net.layer4, 'backward': net.fc}

    elif options.model == 'densenet':
        net = densenet.densenet121(pretrained=True)
        net.classifier = nn.Linear(net.classifier.in_features, out_features=options.num_classes)
        grad_cam_hooks = {'forward': net.features.norm5, 'backward': net.classifier}

    log_string('Model Generated.')
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

    loss = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################

    train_dataset = data(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)
    test_dataset = data(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    evaluate()


    #################################
    # Grad Cam visualizer
    #################################
    visualize(net, test_loader, grad_cam_hooks)
