require 'torch'
require 'nn'
require 'image'

local saliencyMaps = torch.class('saliencyMaps')


function saliencyMaps:compute_map(net, x, y)
  local size = x:size()
  x = torch.reshape(x, 1, size[1], size[2], size[3])
  local scores = net:forward(x)
  local dout = torch.zeros(scores:size()[1])
  dout[y] = 1
  local dx = net:backward(x,dout)
  local map = torch.abs(torch.max(dx, 1):double())
  map = map / torch.max(map)
  map = torch.reshape(map, size[1], size[2], size[3])
  return map
end
