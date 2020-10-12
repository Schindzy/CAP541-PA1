import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode, debug=False):
        super(ConvNet, self).__init__()
        self.debug = debug
        
        # Define various layers here, such as in the tutorial example
        self.conv1 = nn.Conv2d(1, 40, (5,5), (1,1))
        self.conv2 = nn.Conv2d(40, 49, (5, 5), (1, 1)) #49*4*4 = 784

        #If mode is equal to 5, fc layers have 1000 neurons and uses dropout
        if(mode == 5):
            self.fc1 = nn.Linear(784, 1000)
            self.fc2 = nn.Linear(1000, 10)
            self.dropout = nn.Dropout2d(0.5)
        else:
            self.fc1 = nn.Linear(784, 100)
            self.fc2 = nn.Linear(100, 10)
        
        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, x):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        if self.debug:
            print("x:\t\t", x.shape)

        x = torch.flatten(x, start_dim=1)
        if self.debug:
            print("flatten:\t", x.shape)

        x = self.fc1(x)
        x = torch.sigmoid(x)
        if self.debug:
            print("fc1:\t\t", x.shape)

        return x

    # Use two convolutional layers.
    def model_2(self, x):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        x = self.conv1(x)
        x = torch.sigmoid(x)
        if self.debug:
            print("conv1:\t\t", x.shape)

        x = F.max_pool2d(x, 2)
        if self.debug:
            print("max_pool_conv1:\t", x.shape)

        x = self.conv2(x)
        x = torch.sigmoid(x)
        if self.debug:
            print("conv2:\t\t", x.shape)

        x = F.max_pool2d(x, 2)
        if self.debug:
            print("max_pool_conv2:\t", x.shape)

        x = torch.flatten(x, start_dim=1)
        if self.debug:
            print("flatten:\t", x.shape)

        x = self.fc1(x)
        x = torch.sigmoid(x)
        if self.debug:
            print("fc1:\t\t", x.shape)

        return x

    # Replace sigmoid with ReLU.
    def model_3(self, x):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        x = self.conv1(x)
        x = torch.relu(x)
        if self.debug:
            print("conv1:\t\t", x.shape)

        x = F.max_pool2d(x, 2)
        if self.debug:
            print("max_pool_conv1:\t", x.shape)

        x = self.conv2(x)
        x = torch.relu(x)
        if self.debug:
            print("conv2:\t\t", x.shape)

        x = F.max_pool2d(x, 2)
        if self.debug:
            print("max_pool_conv2:\t", x.shape)

        x = torch.flatten(x, start_dim=1)
        if self.debug:
            print("flatten:\t", x.shape)

        x = self.fc1(x)
        x = torch.relu(x)
        if self.debug:
            print("fc1:\t\t", x.shape)

        return x

    # Add one extra fully connected layer.
    def model_4(self, x):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        x = self.conv1(x)
        x = torch.relu(x)
        if self.debug:
            print("conv1:\t\t", x.shape)

        x = F.max_pool2d(x, 2)
        if self.debug:
            print("max_pool_conv1:\t", x.shape)

        x = self.conv2(x)
        x = torch.relu(x)
        if self.debug:
            print("conv2:\t\t", x.shape)

        x = F.max_pool2d(x, 2)
        if self.debug:
            print("max_pool_conv2:\t", x.shape)

        x = torch.flatten(x, start_dim=1)
        if self.debug:
            print("flatten:\t", x.shape)

        x = self.fc1(x)
        x = torch.relu(x)
        if self.debug:
            print("fc1:\t\t", x.shape)

        x = self.fc2(x)
        x = torch.relu(x)
        if self.debug:
            print("fc2:\t\t", x.shape)

        return x

    # Use Dropout now.
    def model_5(self, x):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        x = self.conv1(x)
        x = torch.relu(x)
        if self.debug:
            print("conv1:\t\t", x.shape)

        x = F.max_pool2d(x, 2)
        if self.debug:
            print("max_pool_conv1:\t", x.shape)

        x = self.conv2(x)
        x = torch.relu(x)
        if self.debug:
            print("conv2:\t\t", x.shape)

        x = F.max_pool2d(x, 2)
        if self.debug:
            print("max_pool_conv2:\t", x.shape)

        x = self.dropout(x)
        if self.debug:
            print("dropout:\t", x.shape)

        x = torch.flatten(x, start_dim=1)
        if self.debug:
            print("flatten:\t", x.shape)

        x = self.fc1(x)
        x = torch.relu(x)
        if self.debug:
            print("fc1:\t\t", x.shape)

        x = self.fc2(x)
        x = torch.relu(x)
        if self.debug:
            print("fc2:\t\t", x.shape)

        return x
    
    
