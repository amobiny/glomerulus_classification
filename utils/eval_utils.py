import torch


def compute_accuracy(target, output):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size]
    :param output: Tensor of model predictions of size [batch_size, num_classes]
    :return: prediction accuracy
    """
    num_samples = target.size(0)
    num_correct = torch.sum(target == torch.argmax(output, dim=1))
    accuracy = num_correct.float() / num_samples
    return accuracy
