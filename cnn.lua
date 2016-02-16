----------------------------------------------------------------------
-- CNN implementation using torch.
--
--
-- Martina Marek
----------------------------------------------------------------------


require 'torch'
require 'nn'
require 'cunn';
require 'optim' -- for various trainer methods


-------------------- Parameters for network --------------------------

-- Hyperparameters
learning_rate = 0.001
num_epochs = 5

-- network architecture
fc_layers = [120] -- number of nodes in each fully connected layers (output layer is added additionally)
conv_layers = [10, 20] -- number of nodes in each convolutional layer
filter_size = 5 -- filter size for convolutional layers
pool_size = 2


-- class 0 means empty parking spot, 1 means occupied spot
classes = {'0','1'}

input_channels = 3
-- width = ?
-- height = ?

-------------------- Set up of network ------------------------------

net = nn.Sequential()

-- Convolution on a 3 channel image, with 10 nodes and 5x5 convolutions. Followed by a ReLu and a 2x2 max pooling layer.
net:add(nn.SpatialConvolution(input_channels, conv_layers[1], filter_size, filter_size))  
net:add(nn.ReLU())                       
net:add(nn.SpatialMaxPooling(pool_size,pool_size,pool_size,pool_size))     

-- Convolution with 20 nodes and 5x5 convolutions. Followed by a ReLu and a 2x2 max pooling layer.
net:add(nn.SpatialConvolution(conv_layers[1], conv_layers[2], filter_size, filter_size))
net:add(nn.ReLU())                       
net:add(nn.SpatialMaxPooling(pool_size,pool_size,pool_size,pool_size))

-- Start of fully-connected part: 
--TO DO: size of fully connected layer has to be adjusted to input size!!
net:add(nn.View(16*5*5))                    
net:add(nn.Linear(16*5*5, fc_layers[1]))             
net:add(nn.ReLU())                       
net:add(nn.Linear(fc_layers[1], #classes))
net:add(nn.LogSoftMax())    

-- Add a negative log-likelihood criterion for multi-class classification
criterion = nn.ClassNLLCriterion()

-- Add a trainer with learning rate and number of epochs
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = learning_rate
trainer.maxIteration = num_epochs

-- TO DO: Load the data.

-- Preprocessing of the data
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,#input_channels do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i]) -- for debugging
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    -- Maybe not necessary according to Andrey - we can figure out what works best in the end, but I think in the
    -- assigments we did not subtract the std
    -- stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    -- print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i]) --for debugging
    -- trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


-- Modifications to be able to use a GPU
-- net = net:cuda()
-- criterion = criterion:cuda()
-- trainset.data = trainset.data:cuda()



-- train the network
trainer:train(trainset)

