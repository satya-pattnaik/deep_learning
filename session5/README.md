# Session 5 - Solution

## Intorduction
<img width="370" alt="Screenshot 2023-06-03 at 1 48 00 AM" src="https://github.com/satya-pattnaik/deep_learning/assets/22102468/18d52d52-354d-480c-92f1-d90a040e74b7">

This repo has the solution to Session 5 - **Pytorch 101** from **https://theschoolof.ai/**.

The contents of the Session were as follows:
- Introduction to PyTorch
- PyTorch vs TensorFlow
- Tensors
- AutoGrad
- Tensors and Numpy
- Sharing Memory for Performance
- Tensor Indexing
- Squeezing and Unsqueezing and Other Operations
- Writing a NN from scratch in Pytorch:
- Working with dataset
- Building a network
- Accessing the weights
- Forward Function
- Training
- Loss
- Gradients
- Weight Updates
- Batch Training
- Epochs


## Code Structure
The code is divided into the following strcture
- **model.py**
- **utils.py**
- **S5.ipynb**

### model.py
This file contains the code **block 7**.
It has the definition of the model or the Neural Net.
```
class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256*4*4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) 
        x = x.view(-1, 256*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

### utils.py
- training_data_transformer : Utility function to transform training data.
- test_data_transformer : Utility function to transform training data.
- plot_images_in_batch : Utility function to plot the data samples in batches
- get_losses : Utility function to get losses and accuracy for the current session.
- GetCorrectPredCount : Utility function to estimate training accuracy and loss.
- train - Utilty function to train the model.
- test - Utilty function to test the model.

<img width="617" alt="s5_data" src="https://github.com/satya-pattnaik/deep_learning/assets/22102468/2f00c006-f574-454c-942e-177101b3dcf5">



<img width="640" alt="s5_model_summary" src="https://github.com/satya-pattnaik/deep_learning/assets/22102468/ff704fd4-f268-44ed-b220-09288bed23ee">
<img width="1264" alt="s5_model_loss_plot" src="https://github.com/satya-pattnaik/deep_learning/assets/22102468/d707c7ea-a636-4fa5-bc79-9431b7b235e3">
<img width="1262" alt="s5_model_accuracy_plot" src="https://github.com/satya-pattnaik/deep_learning/assets/22102468/044f5b57-17b3-4801-9f6a-2dc216520025">
