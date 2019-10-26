from optparse import OptionParser


parser = OptionParser()

parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                  help='number of epochs (default: 80)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=100, type='int',
                  help='batch size (default: 16)')
parser.add_option('--df', '--disp_freq', dest='disp_freq', default=10, type='int',
                  help='frequency of displaying the training results (default: 100)')
parser.add_option('--vf', '--val_freq', dest='val_freq', default=34, type='int',
                  help='run validation for each <val_freq> iterations (default: 2000)')
parser.add_option('-j', '--workers', dest='workers', default=0, type='int',
                  help='number of data loading workers (default: 16)')

parser.add_option('--ih', '--img_h', dest='img_h', default=256, type='int',
                  help='input image height (default: 256)')
parser.add_option('--iw', '--img_w', dest='img_w', default=256, type='int',
                  help='input image width (default: 256)')
parser.add_option('--ic', '--img_c', dest='img_c', default=3, type='int',
                  help='number of input channels (default: 3)')
parser.add_option('--nc', '--num_classes', dest='num_classes', default=5, type='int',
                  help='number of classes (default: 5)')

parser.add_option('--m', '--model', dest='model', default='densenet',
                  help='resnet, densenet (default: resnet)')
parser.add_option('--sd', '--save-dir', dest='save_dir', default='./save',
                  help='saving directory of .ckpt models (default: ./save)')

# For CapsNet
parser.add_option('--f1', '--f1', dest='f1', default=256, type='int',
                  help='number of filters for the conv1 layer (default: 256)')
parser.add_option('--k1', '--k1', dest='k1', default=9, type='int',
                  help='filter size of the conv1 layer (default: 9)')

parser.add_option('--f2', '--f2', dest='f2', default=256, type='int',
                  help='number of filters for the primary capsule layer (default: 256)')
parser.add_option('--k2', '--k2', dest='k2', default=9, type='int',
                  help='filter size of the primary capsule layer (default: 9)')

parser.add_option('--pcd', '--primary_cap_dim', dest='primary_cap_dim', default=8, type='int',
                  help='dimension of each primary capsule (default: 8)')
parser.add_option('--dcd', '--digit_cap_dim', dest='digit_cap_dim', default=16, type='int',
                  help='dimension of each digit capsule (default: 16)')

parser.add_option('--lp', '--load_model_path', dest='load_model_path',
                  default='/home/cougarnet.uh.edu/amobiny/Desktop/capsule_network_pytorch/save/20191023_115602/models/82800.ckpt',
                  help='path to load a .ckpt model')


options, _ = parser.parse_args()

