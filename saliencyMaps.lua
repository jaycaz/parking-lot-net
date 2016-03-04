require 'torch'
require 'nn'
require 'image'

local saliencyMaps = torch.class('saliencyMaps')


function saliencyMaps:compute_map(net, x, y)
  local scores = net:forward(x)
  local dout = torch.zeros(scores:size()[1])
  dout[y] = 1
  local dx = net:backward(x,dout)
  local map = torch.abs(torch.max(dx, 1):double())
  --image.save('test.jpg', map)
  return map
end
