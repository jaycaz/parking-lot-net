-- Jordan Cazamias
-- read_data: utilities for reading in PKLot images

-- TODO: Change 'PUC' folder in segmented to 'PUCPR'

require 'torch'
require 'image'
require 'pl'

data_dir = '/home/jordan/Documents/PKLot'
IMG_WIDTH = 49
IMG_HEIGHT = 64

lot_names = {'PUC', 'PUCPR', 'UFPR04', 'UFPR05'}
weather_names = {'Cloudy', 'Rainy', 'Sunny'}

-- TODO: Gets files sequentially for now.  Change so it draws randomly
function get_train_test_sets(num_train, num_test)
local segdir = path.join(path.abspath(data_dir), 'PKLotSegmented')
  local trainset = {}
  trainset.names = {}
  trainset.data = torch.zeros(num_train, 3, IMG_HEIGHT, IMG_WIDTH)
  
  local testset = {}
  testset.names = {}
  testset.data = torch.zeros(num_test, 3, IMG_HEIGHT, IMG_WIDTH)
  
  local n = 1
  
  for lotdir in paths.iterdirs(segdir) do
    for weatherdir in paths.iterdirs(path.join(segdir, lotdir)) do
      for datedir in paths.iterdirs(path.join(segdir, lotdir, weatherdir)) do
        for vacancydir in paths.iterdirs(path.join(segdir, lotdir, weatherdir, datedir)) do
          for file in paths.iterfiles(path.join(segdir, lotdir, weatherdir, datedir, vacancydir)) do
            local filepath = path.join(segdir, lotdir, weatherdir, datedir, vacancydir, file)
            local scaledimage = image.scale(image.load(filepath), IMG_WIDTH, IMG_HEIGHT)
            
            -- Add name to names and data to data tensor
            if n <= num_train then
              trainset.names[n-num_train] = filepath
              trainset.data[{n, {}, {}, {}}]:add(scaledimage)
            elseif n <= num_train + num_test then
              testset.names[n-num_train] = filepath
              testset.data[{n-num_train, {}, {}, {}}]:add(scaledimage)
            else
              break
            end
            n = n+1
          end
        end
      end    
    end
  end
  
  return trainset, testset
end

-- Compiles giant list of all filenames.  Does not load them into memory, only compiles the paths.
-- function get_all_segmented_data(data_dir)
--   local segdir = path.join(path.abspath(data_dir), 'PKLotSegmented')
--   local allfiles = {}
--   local n = 1
--   
--   for lotdir in paths.iterdirs(segdir) do
--     for weatherdir in paths.iterdirs(path.join(segdir, lotdir)) do
--       for datedir in paths.iterdirs(path.join(segdir, lotdir, weatherdir)) do
--         for vacancydir in paths.iterdirs(path.join(segdir, lotdir, weatherdir, datedir)) do
--           for file in paths.iterfiles(path.join(segdir, lotdir, weatherdir, datedir, vacancydir)) do
--             allfiles[n] = path.join(segdir, lotdir, weatherdir, datedir, vacancydir)
--             n = n+1
--             if n%1000 == 0 then
--               print("Added image " .. n)
--             end
--           end
--         end
--       end    
--     end
--   end
--   
--   return allfiles
-- end

trainset, testset = get_train_test_sets(5, 5)

print('Train data: \n', trainset.data:size())
print('Test data: \n', testset.data:size())

-- for i = 1, 10 do
--   print(trainfiles[i])
-- end
-- for i = 1, 10 do
--   print(testfiles[i])
-- end