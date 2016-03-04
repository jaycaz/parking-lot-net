-- modified from https://github.com/karpathy/neuraltalk2/blob/master/misc/DataLoader.lua

require 'hdf5'

local DataLoader = torch.class('DataLoader')


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
  self.label_name = opt.labels
  self.max_spots = opt.max_spots

  self.split_ix = {}
  if opt.weather_cond1 == 'nil' then 
    -- separate out indexes for each of the provided splits
    self.split_ix['train'] = torch.randperm(torch.floor(self.num_images * 0.7))
    local offset = self.split_ix['train']:size()[1]
    self.split_ix['val'] = torch.randperm(torch.floor(self.num_images * 0.2)) + offset
    offset = offset + self.split_ix['val']:size()[1]
    self.split_ix['test'] = torch.randperm(torch.floor(self.num_images - offset)) + offset
    assert((self.split_ix['train']:size()[1] + self.split_ix['val']:size()[1] + self.split_ix['test']:size()[1]) == self.num_images, 'number of images in train/val/test do not match number of images')
  else
    local cond_no1, cond_no2 = resolve_weather_cond(opt.weather_cond1, opt.weather_cond2)
    count_cond1 = 0
    count_cond2 = 0
    for i=1,self.num_images do
      cond = self.h5_file:read('/meta_weather'):partial({i,i})[1]
      if cond == cond_no1 then
        count_cond1 = count_cond1 + 1
      end
      if cond == cond_no2 then
        count_cond2 = count_cond2 + 1
      end
    end
    print(opt.weather_cond1, count_cond1, opt.weather_cond2, count_cond2)
    count = {count_cond1, torch.floor(count_cond2 * 0.7), count_cond2 - torch.floor(count_cond2 * 0.7)}
    local split = {}
    split['train'] = torch.zeros(count[1])
    split['val'] = torch.zeros(count[2])
    split['test'] = torch.zeros(count[3])

    self.num_images = count[1] + count[2] + count[3]
    
    local idx_train = 0
    local idx_val = 0
    local idx_test = 0
    for i=1,self.num_images do
      cond = self.h5_file:read('/meta_weather'):partial({i,i})[1]
      if cond == cond_no1 then
        idx_train = idx_train + 1
        split['train'][idx_train] = i
      end
      if cond == cond_no2 then
        if idx_val < split['val']:size()[1] then
          idx_val = idx_val + 1
          split['val'][idx_val] = i
        else    
          idx_test = idx_test + 1
          split['test'][idx_test] = i
        end
      end
    end
    
    sets = {'train', 'val', 'test'}
    for i=1,#sets do
      self.split_ix[sets[i]] = torch.zeros(count[i])
      local perm = torch.randperm(count[i])
      for j=1,count[i] do
        self.split_ix[sets[i]][j] = split[sets[i]][perm[j]]
      end
    end
  end
  
  -- for debugging
  --print(self.split_ix['train']:size()[1], self.split_ix['val']:size()[1], self.split_ix['test']:size()[1])
  --print(torch.max(self.split_ix['train']))
  
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
     
    label = self.h5_file:read('/' .. self.label_name):partial({ix,ix})
    if self.max_spots ~= 0 then
      if label > self.max_spots then
        label = self.max_spots
      end
    end
    label_batch[i] = label
  end

  -- subtract mean
  for i=1,self.num_channels do
    img_batch_raw[{ {}, {i}, {}, {}  }]:add(-self.mean[i]) -- mean subtraction
  end

  return img_batch_raw, label_batch
end


function resolve_weather_cond(cond1, cond2) 
  local no1 = 0
  local no2 = 0
  if cond1 == 'sunny' then
    no1 = 1
  elseif cond1 == 'cloudy' then
    no1 = 2
  elseif cond1 == 'rainy' then
    no1 = 3
  end
  if cond2 == 'sunny' then
    no2 = 1
  elseif cond2 == 'cloudy' then
    no2 = 2
  elseif cond2 == 'rainy' then
    no2 = 3
  end
  assert(no1 ~= 0 and no2 ~= 0, 'Weather conditions could not be resolved to labels')
  return no1, no2
end
