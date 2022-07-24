"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision


class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################

        # pass
        self.alexnet = torchvision.models.alexnet(pretrained=True).features # output N*256*6*6
        self.model = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic'),  #256*12*12
                                   nn.Dropout(0.2),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 128, 1),                          #128*12*12
                                   nn.Upsample(scale_factor=2, mode='bicubic'),  #128*24*24
                                   nn.Dropout(0.3),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 64, 1),                          #64*24*24
                                   nn.Upsample(scale_factor=2, mode='bicubic'),  #64*48*48
                                   nn.Dropout(0.4),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, num_classes, 1),                  #23*48*48
                                   nn.Upsample(scale_factor=5, mode='bicubic'),  # 23*240*240
                                   nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1)
                                   )
        # self.model = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Conv2d(256, 4096, kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Conv2d(4096, num_classes, kernel_size=1, padding=0),
        #     nn.Upsample(scale_factor=40),
        #     nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1)
        # )
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        # pass
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.alexnet(x)
        x = self.model(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
