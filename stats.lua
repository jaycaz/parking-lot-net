require 'torch'
require 'math'
require 'DataLoader'

stats = {}

-- Runs forward pass of <model> on <data> and returns accuracy
-- based on groud truth labels <labels>
function stats.acc(guesses, labels, num_labels)
  -- Find number of generated labels that match ground truth labels
  data_size = labels:size(1)
  --num_labels = labels:size(2)

  correct = torch.sum(torch.eq(guesses, labels:long()))
  --num_labels = labels:size(1)

  --for i = 1, labels:size(1) do
    --print(string.format("Guess %d, Correct %d", guesses[i], labels[i]))
  --end

  return correct / data_size

end

-- Returns necessary values for a confusion matrix
-- Occupied counts as positive
function stats.confusion(guesses, labels, num_labels)
  data_size = labels:size(1)
  --num_labels = labels:size(2)

  tp = torch.sum(torch.cmul(torch.eq(guesses,2), torch.eq(labels:long(),2)))
  fp = torch.sum(torch.cmul(torch.eq(guesses,2), torch.eq(labels:long(),1)))
  tn = torch.sum(torch.cmul(torch.eq(guesses,1), torch.eq(labels:long(),1)))
  fn = torch.sum(torch.cmul(torch.eq(guesses,1), torch.eq(labels:long(),2)))

  pn_total = tp + fp + tn + fn
  assert(pn_total == data_size, string.format('Confusion matrix parameters (%d) do not add up to total data size (%d)', pn_total, data_size))

  res = {true_pos = tp, false_pos = fp, true_neg = tn, false_neg = fn}
  for k,v in pairs(res) do
    res[k] = res[k] / data_size
  end

  return res
end


-- Returns file paths of all misclassified images
function stats.misclassified(guesses, paths, labels, num_labels)
  misclass_paths = {}

  for i = 1, guesses:size(1) do
    if guesses[i] ~= labels[i] then
      table.insert(misclass_paths, paths[i])
    end
  end

  return misclass_paths
end


-- For counter CNN, get histogram of ground truth vs. choices
-- row is what the gt was, col is what the model guessed
function stats.confusion_counter(guesses, labels, num_labels)
  data_size = labels:size(1)
  --num_labels = labels:size(2)

  confusion_counts = {}
  for i = 1, num_labels do
    confusion_counts[i] = {}
    for j = 1, num_labels do
      confusion_counts[i][j] = 0
    end
  end  

  for i = 1, data_size do
    local gt = labels[i]
    local guess = guesses[i]
    confusion_counts[gt][guess] = confusion_counts[gt][guess] + 1
  end

  return confusion_counts
  
end

--For counter CNN, get histogram of errors
function stats.error_matrix(guesses, labels, num_labels)

  local counts = true

  --local score0 = model:forward(data:narrow(1,1,1))
  --local data_size = data:size(1)
  --local num_labels = score0:size(1)

  local error_counts = {}

  for i = -num_labels, num_labels do
    error_counts[i] = 0
  end

  for i = 1, data_size do
    local gt = labels[i]
    local guess = guesses[i]
    local err = gt - guess
    error_counts[err] = error_counts[err] + 1
  end

  local error_props = {} 
  for k,v in pairs(error_counts) do
    error_props[k] = v / data_size
  end

  if counts then
    return error_counts
  else
    return error_props
  end
  
end

function stats.classify(model, loader, split)
  local BATCH_SIZE = 100

  local data_size = data_size(loader, split)

  data0, _ = loader:getBatch{batch_size = 1, split = split}
  local score0 = model:forward(data0:narrow(1,1,1))
  local num_labels = score0:size(1)

  local scores = torch.Tensor(data_size, num_labels)
  local gt = torch.Tensor(data_size)

  -- Process data in batches
  local i = 1
  while i < data_size do
    local b = math.min(BATCH_SIZE, data_size - i + 1)
    local batch_data, batch_data_y = loader:getBatch{batch_size = b, split = split}
    local batch_scores = model:forward(batch_data)
    
    --print("b: " .. tostring(b))
    --print(batch_scores:size())
    --print(scores:size())
    scores[{{i,i+b-1}}] = batch_scores
    gt[{{i,i+b-1}}] = batch_data_y
    i = i+b
  end

  -- Find number of generated labels that match ground truth labels
  local maxs, guesses = torch.max(scores, 2)
  guesses = torch.reshape(guesses, guesses:size(1))

  --print("Guesses:")
  --print(guesses:size())
  --print("Ground Truth:")
  --print(gt:size())

  return guesses, gt, num_labels
end

function data_size(loader, split)
  local size = -1
  if split == 'train' then
    size = loader:getTrainSize()
  elseif split == 'val' then
    size = loader:getValSize()
  elseif split == 'test' then
    size = loader:getTestSize()
  end
  assert(size > 0, "Data size cannot be <= 0")

  return size
end

return stats


