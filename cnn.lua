----------------------------------------------------------------------
-- CNN implementation using torch.
--
--
-- Martina Marek
----------------------------------------------------------------------


require 'torch'
require 'nn'
require 'optim' -- for various trainer methods
require 'image'
require 'pl'
read_data = require("read_data")

-------------------- Parameters for network --------------------------

local cmd = torch.CmdLine()

-- Hyperparameters
cmd:option('-learning_rate', 0.001)
cmd:option('-num_epochs', 25)
cmd:option('-opt_method', 'sgd')

-- network architecture
fc_layers = {120, 50, 10} -- number of nodes in each fully connected layers (output layer is added additionally)
conv_layers = {10, 20} -- number of nodes in each convolutional layer
filter_size = 5 -- filter size for convolutional layers
pad = 2
stride = 1
pool_size = 2


-- class 0 means empty parking spot, 1 means occupied spot
classes = {'Empty', 'Occupied'}

input_channels = 3
local IMG_WIDTH = 48
local IMG_HEIGHT = 64

-- GPU options
cmd:option('-gpu', 0)

-- TODO: Load the data.
cmd:option('-path', '/Users/martina/Documents/Uni/USA/Stanford/2.Quarter/CNN/Finalproject/PKLot')
--local data_dir = '/home/jordan/Documents/PKLot'

local params = cmd:parse(arg)

if params.gpu > 0 then  
  require 'cunn';
  net = net:cuda()
  criterion = criterion:cuda()
  trainset.data = trainset.data:cuda()
end

NUM_TRAIN = 100
NUM_TEST = 100
trainset, testset = read_data.get_train_test_sets(NUM_TRAIN, NUM_TEST, params.path)
--print(#trainset.label)
-- for k,v in ipairs(trainset.label) do print(v) end

-- Add index operator for trainset
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
 
function trainset:size() 
    return self.data:size(1) 
end

--print(trainset[33][1]:size())

-------------------- Set up of network ------------------------------

net = nn.Sequential()

-- Adding first layer
net:add(nn.SpatialConvolution(input_channels, conv_layers[1], filter_size, filter_size, stride, stride, pad, pad))  
net:add(nn.ReLU())                       
net:add(nn.SpatialMaxPooling(pool_size,pool_size,pool_size,pool_size))     

-- adding rest of conv layers
for i=2,#conv_layers do
  net:add(nn.SpatialConvolution(conv_layers[i - 1], conv_layers[i], filter_size, filter_size, stride, stride, pad, pad))
  net:add(nn.ReLU())                       
  net:add(nn.SpatialMaxPooling(pool_size,pool_size,pool_size,pool_size))
end

-- Start of fully-connected part: 
-- TODO: Not hardcode image size
local pow = #conv_layers
local fcin = {conv_layers[#conv_layers], IMG_HEIGHT/(math.pow(2, pow)), IMG_WIDTH/(math.pow(2, pow))}
local fcinprod = torch.prod(torch.Tensor(fcin))

-- Adding fully connected layers
net:add(nn.View(fcinprod))
-- net:add(nn.Reshape(fcinprod))
net:add(nn.Linear(fcinprod, fc_layers[1]))             
net:add(nn.ReLU())                       

for i=2,#fc_layers do
  net:add(nn.Linear(fc_layers[i - 1], fc_layers[i]))
  net:add(nn.ReLU())
end

net:add(nn.Linear(fc_layers[#fc_layers], #classes))
net:add(nn.LogSoftMax())    

-- Add a negative log-likelihood criterion for multi-class classification
criterion = nn.ClassNLLCriterion()

-- Add a trainer with learning rate and number of epochs
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = params.learning_rate
trainer.maxIteration = params.num_epochs


-- Preprocessing of the data
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,input_channels do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i]) -- for debugging
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    -- Maybe not necessary according to Andrey - we can figure out what works best in the end, but I think in the
    -- assigments we did not subtract the std
    -- stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    -- print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i]) --for debugging
    -- trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

local weights, grad_params = net:getParameters()

-- function for the optim methods
local function f(w)
  assert(w == params)
  grad_params:zero()
  
  -- DOTO: get minibatch of data, convert to cuda

  local scores = model:forward(x)
  local loss = crit:forward(scores, y) --maybe have to reshape scores?!
  local grad_scores = crit:backward(scores, y)
  model:backward(x, grad_scores)
  return loss, grad_params
end

-- train the network
if params.opt_method == 'sgd':
  trainer:train(trainset)
elseif params.opt_method == 'adam': 
  local epoch = math.floor(i / num_train) + 1
  
  -- Maybe decay learning rate
  if epoch % opt.lr_decay_every == 0 then
    local old_lr = optim_config.learningRate
    optim_config = {learningRate = old_lr * opt.lr_decay_factor}
  end
end
