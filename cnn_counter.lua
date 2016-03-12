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
require 'os'
stats = require("stats")
-------------------- Parameters for network --------------------------

local cmd = torch.CmdLine()

-- Hyperparameters
cmd:option('-learning_rate', 0.000737) -- value obtained through crossval
cmd:option('-num_epochs', 1)
cmd:option('-opt_method', 'adam')
cmd:option('-lr_decay_every', 5) -- value obtained through crossval
cmd:option('-lr_decay_factor', 0.95) -- value obtained through crossval
--cmd:option('-momentum', 0.9)
cmd:option('-momentum', 0.0)
cmd:option('-batch_size', 100) -- value obtained through crossval
cmd:option('-batch_norm', 0)

-- Output options
cmd:option('-print_every', 1)
cmd:option('-print_test', 0)
cmd:option('-print_misclassified', 0)
cmd:option('-save_model', 0)
cmd:option('-print_confusion', 0)
cmd:option('-print_error_hist', 0)

-- network architecture
cmd:option('-filter_size', 5)
cmd:option('-pool_size', 2)
--cmd:option('-conv_layers', "10, 20, 40, 80, 160")
--cmd:option('-fc_layers', "60, 40, 20")
cmd:option('-conv_layers', "10, 20, 30")
cmd:option('-fc_layers', "30, 20, 10")
local stride = 1

-- special training mode options (e.g. train only on rainy, test on sunny pictures,...)
cmd:option('-train_set', 'nil')
cmd:option('-test_set', 'nil')
 
local input_channels = 3
local IMG_WIDTH = 256
local IMG_HEIGHT = 128

-- GPU options
cmd:option('-gpu', 0)

-- TODO: Load the data.
cmd:option('-path', '/Users/martina/Documents/Uni/USA/Stanford/2.Quarter/CNN/Finalproject/PKLot')
--local data_dir = '/home/jordan/Documents/PKLot'
cmd:option('-h5_file', '/Users/martina/Documents/Uni/USA/Stanford/2.Quarter/CNN/Finalproject/parking-lot-net/pklot.hdf5')
cmd:option('-labels', 'meta_count_spots')
cmd:option('-max_spots', 20)

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
local loader = DataLoader{h5_file = params.h5_file, train_cond=params.train_set, labels=params.labels, max_spots=params.max_spots}

NUM_TRAIN = loader:getTrainSize()
NUM_TEST = loader:getTestSize()
NUM_VAL = loader:getValSize()


-------------------- Set up of network ------------------------------

assert(#conv_layers < 7, 'ConvNet cannot have more than 6 convolutional layers - image size too small!')
net = nn.Sequential()

local pad = (params.filter_size - 1)/2

-- Adding first layer
net:add(nn.SpatialConvolution(input_channels, conv_layers[1], params.filter_size, params.filter_size, stride, stride, pad, pad))  
if params.batch_norm > 0 then
  net:add(nn.SpatialBatchNormalization(conv_layers[1]))
end
net:add(nn.ReLU())                       
net:add(nn.SpatialMaxPooling(params.pool_size,params.pool_size,params.pool_size,params.pool_size))     

for i=2,#conv_layers do
  net:add(nn.SpatialConvolution(conv_layers[i - 1], conv_layers[i], params.filter_size, params.filter_size, stride, stride, pad, pad))
  if params.batch_norm > 0 then
    net:add(nn.SpatialBatchNormalization(conv_layers[i]))
  end
  net:add(nn.ReLU())                       
  net:add(nn.SpatialMaxPooling(params.pool_size,params.pool_size,params.pool_size,params.pool_size))
end

-- Start of fully-connected part: 
local pow = #conv_layers
local fcin = {conv_layers[#conv_layers], IMG_HEIGHT/(math.pow(2, pow)), IMG_WIDTH/(math.pow(2, pow))}
local fcinprod = torch.prod(torch.Tensor(fcin))

-- Adding fully connected layers
net:add(nn.View(fcinprod))
net:add(nn.Linear(fcinprod, fc_layers[1]))             
--if params.batch_norm > 0 then
 -- net:add(nn.BatchNormalization(fc_layers[1]))
--end
net:add(nn.ReLU())                       

for i=2,#fc_layers do
  net:add(nn.Linear(fc_layers[i - 1], fc_layers[i]))
--  if params.batch_norm > 0 then
  --  net:add(nn.BatchNormalization(fc_layers[i]))
 -- end
  net:add(nn.ReLU())
end

net:add(nn.Linear(fc_layers[#fc_layers], params.max_spots+1))
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
  local scores = net:forward(x)
  local loss = criterion:forward(scores, y) 
  
  --local grad_scores = criterion:backward(scores_view, y_view):view(N, -1)
  local grad_scores = criterion:backward(scores, y)
  net:backward(x, grad_scores)

  --require 'saliencyMaps'
  --local maps = saliencyMaps()
  --maps:compute_map(net, x[1], y[1])  
  return loss, grad_params
end

-- For plotting and evaluation
local train_loss_history = {}
--confusion = optim.ConfusionMatrix(classes)

-- File name to save to if save_model == 1
local model_filename = string.format("model_%d", os.time())


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
    --print(optim_sgd.learningRate)
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


  -- Optionally, save model parameters after every epoch
  if params.save_model == 1 and i % NUM_TRAIN == 0 then
    torch.save(model_filename, net:float())
    if params.gpu == 1 then
      net = net:cuda()
    end
    print(string.format("%f: Model parameters saved to: %s", i / NUM_TRAIN, model_filename))
  end

end -- Finished training


-- Print final train and validation statistics
--print(string.format('Running model on train set (%d images)...', NUM_TRAIN))
local train, train_y = loader:getBatch{batch_size = NUM_TRAIN, split = 'train'}
local val, val_y = loader:getBatch{batch_size = NUM_VAL, split = 'val'}
local test, test_y = loader:getBatch{batch_size = NUM_TEST, split = 'test'}

local train_acc = stats.acc(net:double(), train:double(), train_y:int())
local val_acc = stats.acc(net:double(), val:double(), val_y:int())


-- Optionally, print test statistics
local test_acc = -1
if params.print_test == 1 then
  local test, test_y, paths = loader:getBatch{batch_size = NUM_TEST, split = 'test', get_paths=true}
  test_acc = stats.acc(net:double(), test:double(), test_y:int())
end

print("**Train Acc,Val Acc,Test Acc,Learn Rate,Batch Size,LR Decay Rate,LR Decay Every,Weather Train,Weather Test")
print(string.format("*%04f,%04f,%04f,%04f,%d,%04f,%d,%s,%s", 
                    train_acc, val_acc, test_acc, params.learning_rate, params.batch_size, params.lr_decay_factor, 
                    params.lr_decay_every, params.train_set, params.test_set))


  -- Optionally, print all the test files that were incorrectly labeled
if params.print_misclassified == 1 then
  misclass_paths = stats.misclassified(net:double(), test:double(), paths, test_y:int())
  for i, path in ipairs(misclass_paths) do
    print(path)
  end
end

-- Optionally, print confusion matrix parameters for test set
if params.print_confusion == 1 then
  local conf = stats.confusion_counter(net:double(), test:double(), test_y:int())

  print('Confusion Matrix Statistics:')
  for k1,v1 in pairs(conf) do
    for k2,v2 in pairs(v1) do
      print(string.format('%s,%s,%f', k1, k2, v2))
    end
  end
end

-- Optionally, print error histogram
if params.print_error_hist == 1 then
  --local val, val_y = loader:getBatch{batch_size = NUM_VAL, split = 'val'}
  local errs = stats.error_matrix(net:double(), test:double(), test_y:int())

  print('Error Histogram Statistics:')
  for k,v in pairs(errs) do
    print(string.format('%s,%f', k, v))
  end
end


