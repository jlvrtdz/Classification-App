import torch
from torch import nn
from torch.nn import functional as F
import joblib

class ClassifierNN(nn.Module):
    """
    Neural network classifier.

    Parameters
    ----------
    activation_function : torch.nn.Module
        The activation function to be used in the network.

    Attributes
    ----------
    activation_function : torch.nn.Module
        The activation function for the network.
    fcn1 : torch.nn.Linear
        The first fully connected layer with input size 14 and output size 32.
    fcn2 : torch.nn.Linear
        The second fully connected layer with input size 32 and output size 64.
    fcn3 : torch.nn.Linear
        The third fully connected layer with input size 64 and output size 128.
    fcn4 : torch.nn.Linear
        The fourth fully connected layer with input size 128 and output size 2.

    Methods
    -------
    forward(x)
        Forward pass of the neural network.

    """

    def __init__(self):
        super().__init__()
        optimal_activation_function = F.relu

        self.activation_function = optimal_activation_function

        # Define fully connected layers
        self.fcn1 = nn.Linear(14, 32)
        self.fcn2 = nn.Linear(32, 64)
        self.fcn3 = nn.Linear(64, 128)
        self.fcn4 = nn.Linear(128, 2)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (-1, 14).

        Returns
        -------
        torch.Tensor
            Output tensor after the forward pass.
        """
        x = x.view(-1, 14)

        x = self.activation_function(self.fcn1(x))
        x = self.activation_function(self.fcn2(x))
        x = self.activation_function(self.fcn3(x))

        x = self.fcn4(x)

        return x

MLP = ClassifierNN()

MLP.load_state_dict(torch.load(r"Streamlit\mlp_parameters_tensor.pth"))


# Load the pre-trained LGBM
LGBM_path = r"Streamlit\best_model (imbalanced).joblib"
LGBM = joblib.load(LGBM_path, mmap_mode=None)
