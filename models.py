import torch.nn as nn


class CNN(nn.Module):
    # Small 2-D CNN that treats each sample as a 1-channel "image" of
    # 10000 (time) x 12 (space) pixels and classifies it into 6 event types.
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(             # input shape (1, 10000, 12)
            nn.Conv2d(
                in_channels=1,  # one input channel (greyscale)
                out_channels=5,  # number of filters
                kernel_size=(200, 3),  # filter size: 200 time steps x 3 spatial points
                stride=(50, 1),  # filter movement/step
                padding=1,
            ),                                  # -> (5, 197, 12)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1),  # -> (5, 99, 7)
        )
        self.conv2 = nn.Sequential(               # input shape (5, 99, 7)
            nn.Conv2d(5, 10, (20, 2), (4, 1), 1),  # -> (10, 21, 8)
            nn.ReLU(),                             # activation
            nn.MaxPool2d(kernel_size=2),           # -> (10, 10, 4)
        )
        self.out = nn.Linear(10 * 10 * 4, 6)      # fully connected layer, output 6 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # flatten to (batch, 400)
        feature = x                 # the 400-dim feature vector, used for visualisation
        output = self.out(x)        # class scores (logits)
        return feature, output
