require 'torch'
require 'math'

local stats = {}

-- Runs forward pass of <model> on <data> and returns accuracy
-- based on groud truth labels <labels>
function stats.acc(model, data, labels)
  -- Find number of generated labels that match ground truth labels
  local indices = classify(model, data)

  local correct = torch.sum(torch.eq(indices, labels:long()))
  local num_labels = labels:size(1)

  --for i = 1, labels:size(1) do
    --print(string.format("Guess %d, Correct %d", indices[i], labels[i]))
  --end

  return correct / num_labels

end

-- Returns necessary values for a confusion matrix
-- Occupied counts as positive
function stats.confusion(model, data, labels)

  local score0 = model:forward(data:narrow(1,1,1))
  local data_size = data:size(1)
  local num_labels = score0:size(1)

  local indices = classify(model, data)

  local tp = torch.sum(torch.cmul(torch.eq(indices,2), torch.eq(labels:long(),2)))
  local fp = torch.sum(torch.cmul(torch.eq(indices,2), torch.eq(labels:long(),1)))
  local tn = torch.sum(torch.cmul(torch.eq(indices,1), torch.eq(labels:long(),1)))
  local fn = torch.sum(torch.cmul(torch.eq(indices,1), torch.eq(labels:long(),2)))

  pn_total = tp + fp + tn + fn
  assert(pn_total == data_size, string.format('Confusion matrix parameters (%d) do not add up to total data size (%d)', pn_total, data_size))

  local res = {true_pos = tp, false_pos = fp, true_neg = tn, false_neg = fn}
  for k,v in pairs(res) do
    res[k] = res[k] / data_size
  end

  return res
end


-- Returns file paths of all misclassified images
function stats.misclassified(model, data, paths, labels)
  local misclass_paths = {}
  local indices = classify(model, data)

  for i = 1, indices:size(1) do
    if indices[i] ~= labels[i] then
      table.insert(misclass_paths, paths[i])
    end
  end

  return misclass_paths
end


function classify(model, data)
  local BATCH_SIZE = 100
  local score0 = model:forward(data:narrow(1,1,1))
  local data_size = data:size(1)
  local num_labels = score0:size(1)
  local scores = torch.Tensor(data_size, num_labels)

  -- Process data in batches
  local i = 1
  while i < data_size do
    b = math.min(BATCH_SIZE, data_size - i)
    --print(data[{{i,i+b}}]:size())
    scores[{{i,i+b}}] = model:forward(data[{{i,i+b}}])
    i = i+b
  end

  -- Find number of generated labels that match ground truth labels
  maxs, indices = torch.max(scores, 2)
  indices = torch.reshape(indices, indices:size(1))

  return indices
end


-- For counter CNN, get histogram of choices vs ground truth
function stats.confusion_counter(model, data, labels)

  local score0 = model:forward(data:narrow(1,1,1))
  local data_size = data:size(1)
  local num_labels = score0:size(1)

  local indices = classify(model, data)

  local confusion_counts = {}
  for i = 1, num_labels do
    confusion_counts[i] = {}
    for j = 1, num_labels do
      confusion_counts[i][j] = 0
    end
  end  

  for i = 1, data_size do
    local gt = labels[i]
    local guess = indices[i]
    confusion_counts[gt][guess] = confusion_counts[gt][guess] + 1
  end

  return confusion_counts
  
end

--For counter CNN, get histogram of errors
function stats.error_matrix(model, data, labels)

  local counts = true

  local score0 = model:forward(data:narrow(1,1,1))
  local data_size = data:size(1)
  local num_labels = score0:size(1)

  local indices = classify(model, data)

  error_counts = {}

  for i = -num_labels, num_labels do
    error_counts[i] = 0
  end

  for i = 1, data_size do
    local gt = labels[i]
    local guess = indices[i]
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

return stats

