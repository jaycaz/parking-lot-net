require 'torch'

local stats = {}

-- Runs forward pass of <model> on <data> and returns accuracy
-- based on groud truth labels <labels>
function stats.acc(model, data, labels)

  local score0 = model:forward(data[1])

  local num_data = data:size(1)
  local num_labels = score0:size(1)

  scores = torch.Tensor(num_data, num_labels)

  for i = 1, data:size(1) do
    scores[i] = model:forward(data[i])
  end

  maxs, indices = torch.max(scores, 2)

  --print(maxs:size())
  --print(labels:size())

  --print(indices)
  --print(labels)

  --print(torch.eq(indices, labels:long()))
  local correct = torch.sum(torch.eq(indices, labels:long()))
  --print(correct)

  local num_labels = labels:size()[1]

  return correct / num_labels

end

return stats
