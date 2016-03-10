require 'hdf5'
require 'math'

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
  print("max_spots ".. opt.max_spots)

  self.split_ix = {}
  if opt.train_cond == 'nil' then 
    -- separate out indexes for each of the provided splits
    self.split_ix['train'] = torch.randperm(torch.floor(self.num_images * 0.7))
    local offset = self.split_ix['train']:size()[1]
    self.split_ix['val'] = torch.randperm(torch.floor(self.num_images * 0.2)) + offset
    offset = offset + self.split_ix['val']:size()[1]
    self.split_ix['test'] = torch.randperm(torch.floor(self.num_images - offset)) + offset
    assert((self.split_ix['train']:size()[1] + self.split_ix['val']:size()[1] + self.split_ix['test']:size()[1]) == self.num_images, 'number of images in train/val/test do not match number of images')
  else
    self.cond_counts = {}
    local cond_no = resolve_labels(opt.train_cond1)
    local metadata = resolve_metadata(opt.train_cond1)

    for i=1,self.num_images do
      cond = self.h5_file:read(metadata):partial({i,i})[1]
      self.cond_counts[cond] = self.cond_counts[cond] + 1
    end
    local count_cond = self.cond_counts[cond_no]
    count = {torch.floor(count_cond * 0.7), torch.floor(count_cond * 0.2), count_cond - torch.floor(count_cond * 0.7) - torch.floor(count_cond * 0.2)}
    local split = {}
    split['train'] = torch.zeros(count[1])
    split['val'] = torch.zeros(count[2])
    split['test'] = torch.zeros(count[3])

    -- Mark indices that should be added to each set
    local idx_train = 0
    local idx_val = 0
    local idx_test = 0
    for i=1,self.num_images do
      cond = self.h5_file:read(metadata):partial({i,i})[1]
      if cond == cond_no then
        if idx_train < split['train']:size()[1] then
          idx_train = idx_train + 1
          split['train'][idx_train] = i
        elseif idx_val < split['val']:size()[1] then
          idx_val = idx_val + 1
          split['val'][idx_val] = i
        else    
          idx_test = idx_test + 1
          split['test'][idx_test] = i
        end
      end
    end

    self.num_images = count[1] + count[2] + count[3]
    
    sets = {'train', 'val', 'test'}
    idxs = {idx_train, idx_val, idx_test}

    assert(idx_train == count[1] and idx_val == count[2] and idx_test == count[3],
            string.format("Indexes for weather conditions were not counted up correctly:\n(%d, %d, %d) vs. (%d, %d, %d)",
            count[1], count[2], count[3], idx_train, idx_val, idx_test))

    for i=1,#sets do
      self.split_ix[sets[i]] = torch.zeros(count[i])
      local perm = torch.randperm(count[i])
      for j=1,count[i] do
        self.split_ix[sets[i]][j] = split[sets[i]][perm[j]]
      end
    end
  end
  
  -- for debugging
  -- print(self.split_ix['train']:size()[1], self.split_ix['val']:size()[1], self.split_ix['test']:size()[1])
  -- print(torch.max(self.split_ix['train']))
  
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
    local batch_num_it = math.min(i + batch_size, training_size - i)
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


function DataLoader:reloadTestData(split)
  local cond = resolve_labels(split)
  local metadata = resolve_metadata(split)
  count = {torch.floor(cond * 0.2), torch.floor(cond * 0.1)}
  local split = {}
  split['val'] = torch.zeros(count[1])
  split['test'] = torch.zeros(count[2])

  self.num_images = self.split_ix['train']:size()[1] + count[2] + count[3]
   
  local idx_val = 0
  local idx_test = 0
  for i=1,self.num_images do
    c = self.h5_file:read('/meta_weather'):partial({i,i})[1]
    if c == cond then
      if idx_val < split['val']:size()[1] then
        idx_val = idx_val + 1
        split['val'][idx_val] = i
      else   
        idx_test = idx_test + 1
        split['test'][idx_test] = i
      end
    end
  end

  sets = {'val', 'test'}
    for i=1,#sets do
      self.split_ix[sets[i]] = torch.zeros(count[i])
      local perm = torch.randperm(count[i])
      for j=1,count[i] do
        self.split_ix[sets[i]][j] = split[sets[i]][perm[j]]
      end
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
  local paths = {}
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
     
    label = self.h5_file:read('/' .. self.label_name):partial({ix,ix})[1]
   
    -- Optionally store path
    if opt.get_paths == true then
      local year = self.h5_file:read('/meta_year'):partial({ix, ix})[1]
      local month = self.h5_file:read('/meta_month'):partial({ix, ix})[1]
      local day = self.h5_file:read('/meta_day'):partial({ix, ix})[1]
      local hour = self.h5_file:read('/meta_hour'):partial({ix, ix})[1]
      local minute = self.h5_file:read('/meta_minute'):partial({ix, ix})[1]
      local second = self.h5_file:read('/meta_second'):partial({ix, ix})[1]
      local space = self.h5_file:read('/meta_space'):partial({ix, ix})[1]

      local occupied = self.h5_file:read('/meta_occupied'):partial({ix, ix})[1]
      if occupied == 1 then occupied = 'Empty' else occupied = 'Occupied' end
      local lot = self.h5_file:read('/meta_lot'):partial({ix, ix})[1]
      if lot == 1 then lot = 'PUC' elseif lot == 2 then lot = 'UFPR04' else lot = 'UFPR05' end
      local weather = self.h5_file:read('/meta_weather'):partial({ix, ix})[1]
      if weather == 1 then weather = 'Sunny' elseif weather == 2 then weather = 'Cloudy' else weather = 'Rainy' end

      local path = string.format('%s/%s/%04d-%02d-%02d/%s/%04d-%02d-%02d_%02d_%02d_%02d#%03d.jpg',
                    lot, weather, year, month, day, occupied, year, month, day, hour, minute, second, space)
      --print(path)
      table.insert(paths, path)
    end

    if self.max_spots ~= 0 then
      if label > self.max_spots then
        label = self.max_spots
      end
    end
    label_batch[i] = label
  end

  -- subtract mean
  if opt.raw ~= true then
    for i=1,self.num_channels do
      img_batch_raw[{ {}, {i}, {}, {}  }]:add(-self.mean[i]) -- mean subtraction
    end
  end

  if opt.get_paths == true then
    return img_batch_raw, label_batch, paths
  else
    return img_batch_raw, label_batch
  end
end


function resolve_metadata(cond)
  if (cond == 'sunny') or (cond == 'rainy') or (cond == 'cloudy') then
    return '/meta_weather'
  end
  if (cond == 'PUC') or (cond == 'UFPR04') or (cond == 'UFPR05') then
    return '/meta_lot'
  end
  return nil
end



function resolve_labels(cond) 
  local no1 = 0
  if cond == 'sunny' then
    no = 1
  elseif cond == 'cloudy' then
    no = 2
  elseif cond == 'rainy' then
    no = 3
  end
  if cond == 'PUC' then
    no = 1
  elseif cond == 'UFPR04' then
    no = 2
  elseif cond == 'UFPR05' then
    no = 3
  end
  assert(no ~= 0, 'Train set conditions could not be resolved to label')
  return no
end
