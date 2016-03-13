-- Helper script to easily load statistics from pretrained lot counter model

require 'torch'
require 'nn'
require 'DataLoader'
require 'io'
stats = require("stats")

params = {
  h5_file = '/home/jordan/Documents/parking-lot-net/h5/lots.hdf5',
  --h5_file = '/home/jordan/Documents/parking-lot-net/h5/lots-tiny.hdf5',
  model_path = '/home/jordan/Documents/parking-lot-net/pretrained/counter_final',
  train_set = nil,
  labels = 'meta_count_spots',
  max_spots = 20
}

local loader = DataLoader{h5_file = params.h5_file, labels=params.labels, max_spots=params.max_spots}


NUM_TRAIN = loader:getTrainSize()
NUM_TEST = loader:getTestSize()
NUM_VAL = loader:getValSize()

net = torch.load(params.model_path)

print("Pretrained model loaded")

-- Print final train and validation statistics
--print(string.format('Running model on train set (%d images)...', NUM_TRAIN))
--local train, train_y = loader:getBatch{batch_size = NUM_TRAIN, split = 'train'}
--print("Train set loaded")
--local val, val_y = loader:getBatch{batch_size = NUM_VAL, split = 'val'}
--print("Val set loaded")
--local test, test_y = loader:getBatch{batch_size = NUM_TEST, split = 'test'}
--print("Test set loaded")
local train_idx, train_gt, num_labels = stats.classify(net:double(), loader, 'train')
local val_idx, val_gt, _ = stats.classify(net:double(), loader, 'val')
local test_idx, test_gt, _ = stats.classify(net:double(), loader, 'test')

local train_acc = stats.acc(train_idx, train_gt, num_labels)
local val_acc = stats.acc(val_idx, val_gt, num_labels)
local test_acc = stats.acc(test_idx, test_gt, num_labels)

-- Print test statistics
print("**Train Acc,Val Acc,Test Acc")
print(string.format("*%04f,%04f,%04f", train_acc, val_acc, test_acc))


  -- Print all the test files that were incorrectly labeled
--misclass_paths = stats.misclassified(net:double(), test:double(), paths, test_y:int())
--for i, path in ipairs(misclass_paths) do
  --print(path)
--end

-- Print confusion matrix parameters for test set
local conf = stats.gt_vs_guess(test_idx, test_gt, num_labels)

print('Confusion Matrix Statistics:')
for k1,v1 in pairs(conf) do
  print('')
  for k2,v2 in pairs(v1) do
    io.write(string.format("%d,", v2))
  end
end
print('')

-- Print error histogram
  --local val, val_y = loader:getBatch{batch_size = NUM_VAL, split = 'val'}
local errs = stats.error_matrix(test_idx, test_gt, num_labels)

print('Error Histogram Statistics:')
for k,v in pairs(errs) do
  print(string.format('%s,%f', k, v))
end


