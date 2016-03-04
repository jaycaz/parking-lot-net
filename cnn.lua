----------------------------------------------------------------------
-- CNN implementation using torch.
--
-- Used https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua and
-- https://github.com/jcjohnson/torch-rnn/blob/master/train.lua as example code
-- 
-- Martina Marek
----------------------------------------------------------------------


require 'torch'
require 'nn'
require 'optim' -- for various trainer methods
require 'image'
require 'pl'
read_data = require("read_data")
stats = require("stats")
-------------------- Parameters for network --------------------------

local cmd = torch.CmdLine()

-- Hyperparameters
cmd:option('-learning_rate', 0.001)
cmd:option('-num_epochs', 25)
cmd:option('-opt_method', 'sgd')
cmd:option('-lr_decay_every', 5)
cmd:option('-lr_decay_factor', 0.5)
cmd:option('-momentum', 0.9)
cmd:option('-batch_size', 25)

-- Output options
cmd:option('-print_every', 1)
cmd:option('-print_test', 0)

-- network architecture
cmd:option('-filter_size', 5)
cmd:option('-pool_size', 2)
cmd:option('-conv_layers', "10, 20, 30")
cmd:option('-fc_layers', "10, 20, 30")
local stride = 1

-- special training mode options (e.g. train only on rainy, test on sunny pictures,...)
cmd:option('-weather_train', 'nil')
cmd:option('-weather_test', 'nil')
 
-- class 0 means empty parking spot, 1 means occupied spot
classes = {'Empty', 'Occupied'}

local input_channels = 3
local IMG_WIDTH = 48
local IMG_HEIGHT = 64

-- GPU options
cmd:option('-gpu', 0)

-- TODO: Load the data.
cmd:option('-path', '/Users/martina/Documents/Uni/USA/Stanford/2.Quarter/CNN/Finalproject/PKLot')
--local data_dir = '/home/jordan/Documents/PKLot'
cmd:option('-h5_file', '/Users/martina/Documents/Uni/USA/Stanford/2.Quarter/CNN/Finalproject/parking-lot-net/pklot.hdf5')
cmd:option('-labels', 'meta_occupied')
cmd:option('-max_spots', 0)

local params = cmd:parse(arg)

function string:split(sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(c) fields[#fields+1] = tonumber(c) end)
        return fields
end


conv_layers = params.conv_layers:split(", ")
fc_layers = params.fc_layers:split(", ")


require 'DataLoader'
local loader = DataLoader{h5_file = params.h5_file, weather_cond1=params.weather_train, weather_cond2=params.weather_test, labels=params.labels, max_spots=params.max_spots}

NUM_TRAIN = loader:getTrainSize()
NUM_TEST = loader:getTestSize()
NUM_VAL = loader:getValSize()


-------------------- Set up of network ------------------------------

assert(#conv_layers < 5, 'ConvNet cannot have more than 4 convolutional layers - image size too small!')
net = nn.Sequential()

local pad = (params.filter_size - 1)/2

-- Adding first layer
net:add(nn.SpatialConvolution(input_channels, conv_layers[1], params.filter_size, params.filter_size, stride, stride, pad, pad))  
net:add(nn.ReLU())                       
net:add(nn.SpatialMaxPooling(params.pool_size,params.pool_size,params.pool_size,params.pool_size))     

-- adding rest of conv layers
for i=2,#conv_layers do
  net:add(nn.SpatialConvolution(conv_layers[i - 1], conv_layers[i], params.filter_size, params.filter_size, stride, stride, pad, pad))
  net:add(nn.ReLU())                       
  net:add(nn.SpatialMaxPooling(params.pool_size,params.pool_size,params.pool_size,params.pool_size))
end

-- Start of fully-connected part: 
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

-- If GPU, convert to CudaTensors
if params.gpu > 0 then  
  require 'cunn';
  require 'cutorch';
  net = net:cuda()
  criterion = criterion:cuda()
end

weights, grad_params = net:getParameters()


-- function for the optim methods
local function f(w)
  assert(w == weights)
  grad_params:zero()
  
  -- Get minibatch of data, convert to cuda
  local x, y = loader:getBatch{batch_size = params.batch_size, split = 'train'}
  if params.gpu > 0 then
    x = x:cuda()
    y = y:cuda()
  end
  --local x = data.images:double()
  --local y = data.labels:double()
  local scores = net:forward(x)
  --local scores_view = scores:view(N, -1)
  --local y_view = y:view(N)
  --local loss = crit:forward(scores_view, y_view)
  local loss = criterion:forward(scores, y) --maybe have to reshape scores?!
  
  --local grad_scores = criterion:backward(scores_view, y_view):view(N, -1)
  local grad_scores = criterion:backward(scores, y)
  net:backward(x, grad_scores)
  return loss, grad_params
end

-- For plotting and evaluation
local train_loss_history = {}
confusion = optim.ConfusionMatrix(classes)


-- train the network
  local optim_config = {learningRate = params.learning_rate}
  local num_iterations = params.num_epochs * NUM_TRAIN 
  
for i = 1, num_iterations do
  local epoch = math.floor(i / NUM_TRAIN) + 1
    
  -- update step
  local loss = 0
  if params.opt_method == 'sgd' then
    optim_sgd = optim_sgd or {
          learningRate = params.learning_rate,
          momentum = params.momentum,
          learningRateDecay = params.lr_decay_factor
       }
    if new_lr ~= nil then
      optim_sgd.learningRate = new_lr
    end
    print(optim_sgd.learningRate)
    _, loss = optim.sgd(f, weights, optim_sgd)
  elseif params.opt_method == 'adam' then
    optim_adam = optim_adam or {
          learningRate = params.learning_rate,
          learningRateDecay = params.lr_decay_factor
       }
    if new_lr ~= nil then
      optim_adam.learningRate = new_lr
    end
    _, loss = optim.adam(f, weights, optim_adam)
  else
    print('Unkown update method.')
  end

  table.insert(train_loss_history, loss[1])

  -- Maybe decay learning rate
  if epoch % params.lr_decay_every == 0 then
    local old_lr 
    if params.opt_method == 'sgd' then
      old_lr = optim_sgd.learningRate
    elseif params.opt_method == 'adam' then
      old_lr = optim_adam.learningRate
    end
    new_lr = old_lr * params.lr_decay_factor
  end


  -- update confusion
  --confusion:add(output, targets[i])
  --local trainAccuracy = confusion.totalValid * 100
  --confusion:zero()

  -- print
  if params.print_every > 0 and i % params.print_every == 0 then
    local float_epoch = i / NUM_TRAIN -- + 1
    local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
    local args = {msg, float_epoch, params.num_epochs, i, num_iterations, loss[1]}
    print(string.format(unpack(args)))
  end
  
  weights, grad_params = net:getParameters()
end -- Finished training


-- Print final train and validation statistics
print(string.format('Running model on train set (%d images)...', NUM_TRAIN))
local train, train_y = loader:getBatch{batch_size = NUM_VAL, split = 'train'}
local train_acc = stats.acc(net:double(), train:double(), train_y:int())

print(string.format('Train Accuracy: %04f', train_acc))

print(string.format('Running model on validation set (%d images)...', NUM_VAL))

local val, val_y = loader:getBatch{batch_size = NUM_VAL, split = 'val'}
local val_acc = stats.acc(net:double(), val:double(), val_y:int())


print(string.format('Val Accuracy: %04f', val_acc))

print("*Val Acc,Train Acc,Learn Rate,Batch Size,LR Decay Rate,LR Decay Every")
print(string.format("**%04f,%04f,%04f,%d,%04f,%d", 
                    val_acc, train_acc, params.learning_rate, params.batch_size, params.lr_decay_factor, params.lr_decay_every))


-- Optionally, print test statistics
if params.print_test == 1 then
  local test, test_y = loader:getBatch{batch_size = NUM_VAL, split = 'test'}
  local test_acc = stats.acc(net:double(), test:double(), test_y:int())

  print(string.format('Test Accuracy: %04f', test_acc))

  print("*Test Acc,Train Acc,Learn Rate,Batch Size,LR Decay Rate,LR Decay Every")
  print(string.format("**%04f,%04f,%04f,%d,%04f,%d", 
                      test_acc, train_acc, params.learning_rate, params.batch_size, params.lr_decay_factor, params.lr_decay_every))
end
