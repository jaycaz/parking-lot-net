-- modified from https://github.com/karpathy/neuraltalk2/blob/master/misc/DataLoader.lua

require 'hdf5'

local DataLoader = torch.class('DataLoader')


--TO DO: adjust code to our data
function DataLoader:__init(opt)

  -- open the hdf5 file
  print('DataLoader loading h5 file: ', opt)
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  
  -- extract image size from dataset
  local images_size = self.h5_file:read('/data'):dataspaceSize()
  assert(#images_size == 4, '/data should be a 4D tensor')
  self.num_images = images_size[1]
  self.num_channels = images_size[2]
  self.height = images_size[3]
  self.width = images_size[4]
  print(string.format('read %d images of size %dx%dx%d', self.num_images, 
            self.num_channels, self.height, self.width))
  
  -- separate out indexes for each of the provided splits
  self.split_ix = {}
  self.split_ix['train'] = torch.randperm(torch.floor(self.num_images * 0.7))
  local offset = self.split_ix['train']:size()[1]
  self.split_ix['val'] = torch.randperm(torch.floor(self.num_images * 0.2)) + offset
  offset = offset + self.split_ix['val']:size()[1]
  self.split_ix['test'] = torch.randperm(torch.floor(self.num_images - offset)) + offset
  assert((self.split_ix['train']:size()[1] + self.split_ix['val']:size()[1] + self.split_ix['test']:size()[1]) == self.num_images, 'number of images in train/val/test do not match number of images')
  self.iterators = {}
  
  for k,v in pairs(self.split_ix) do
    self.iterators[k] = 1
    print(string.format('assigned %d images to split %s', v:size()[1], k))
  end

  local training_size = self.split_ix['train']:size()[1]
  local batch_size = 1000
  -- preprocessing: subtract the mean for each channel
  local mean = {}
  for i=1,training_size,batch_size do
    local batch_num_it = 0
    if (i + batch_size) > training_size then
      batch_num_it = training_size - i
    else
      batch_num_it = batch_size
    end
    local imgs = torch.zeros(batch_num_it, self.num_channels, self.height, self.width)
    for j=1,batch_num_it do
      ix = self.split_ix['train'][i+j]
      local img = self.h5_file:read('/data'):partial({ix,ix},{1,self.num_channels},
                            {1,self.height},{1,self.width})
      imgs[j] = img
    end
    for j=1,self.num_channels do
      if mean[j] == nil then mean[j] = 0 end
      mean[j] = mean[j] + batch_num_it * imgs[{ {}, {j}, {}, {}  }]:mean() -- mean estimation
    end
  end

  self.mean = {}
  for i=1,self.num_channels do
    self.mean[i] = mean[i] / training_size -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. self.mean[i])

  end
end

function DataLoader:getTrainSize()
  return self.split_ix['train']:size()[1]
end

function DataLoader:getTestSize()
  return self.split_ix['test']:size()[1]
end

function DataLoader:getValSize()
  return self.split_ix['val']:size()[1]
end

-- function for minibatch read in
function DataLoader:getBatch(opt)
  local split = opt.split -- lets require that user passes this in, for safety
  local batch_size = opt.batch_size -- how many images get returned at one time (to go through CNN)

  local split_ix = self.split_ix[split]
  assert(split_ix, 'split ' .. split .. ' not found.')
  
  local img_batch_raw = torch.zeros(batch_size, self.num_channels, self.height, self.width)
  local label_batch = torch.zeros(batch_size)
  local max_index = split_ix:size()[1]
  
  for i=1,batch_size do
    local ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    self.iterators[split] = ri_next

    ix = split_ix[ri]
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)

    -- fetch the image from h5
    local img = self.h5_file:read('/data'):partial({ix,ix},{1,self.num_channels},
                            {1,self.height},{1,self.width})
    img_batch_raw[i] = img
     
    label_batch[i] = self.h5_file:read('/meta_occupied'):partial({ix,ix})
  end

  -- subtract mean
  for i=1,self.num_channels do
    img_batch_raw[{ {}, {i}, {}, {}  }]:add(-self.mean[i]) -- mean subtraction
  end

  return img_batch_raw, label_batch
end
