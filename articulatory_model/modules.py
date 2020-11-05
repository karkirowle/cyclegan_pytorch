import torch.nn as nn
from torch.functional import F


class Modern_DBLSTM_1(nn.Module):
    """
    DBLSTM implementation
    """

    def __init__(self, input_size, hidden_size, hidden_size_2, num_classes):
        super(Modern_DBLSTM_1, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # You could do more layers here, completely optional
        self.num_layers = 2
        self.hidden_dim = hidden_size
        self.lstm1 = nn.LSTM(hidden_size, hidden_size_2, bidirectional=True, num_layers=self.num_layers,
                             batch_first=True)
        # You had to pay attention to the directionality
        # and the last linear layer is important
        self.fc3 = nn.Linear(hidden_size_2 * 2, num_classes)

    def forward(self, x, mask=None):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))

        out, hidden = self.lstm1(out)

        out = self.fc3(out)
        return out
