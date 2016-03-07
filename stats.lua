require 'torch'
require 'math'

local stats = {}

-- Runs forward pass of <model> on <data> and returns accuracy
-- based on groud truth labels <labels>
function stats.acc(model, data, labels)

  local BATCH_SIZE = 1000
  local score0 = model:forward(data[1])

  local data_size = data:size(1)
  local num_labels = score0:size(1)

  scores = torch.Tensor(data_size, num_labels)

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
  local correct = torch.sum(torch.eq(indices, labels:long()))
  local num_labels = labels:size(1)

  --for i = 1, labels:size(1) do
    --print(string.format("Guess %d, Correct %d", indices[i], labels[i]))
  --end

  return correct / num_labels

end

return stats
