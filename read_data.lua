-- Jordan Cazamias
-- read_data: utilities for reading in PKLot images

-- TODO: Change 'PUC' folder in segmented to 'PUCPR'

require 'torch'
require 'pl'

data_dir = '/home/jordan/Documents/PKLot'

lot_names = {'PUC', 'PUCPR', 'UFPR04', 'UFPR05'}
weather_names = {'Cloudy', 'Rainy', 'Sunny'}

-- TODO: Gets files sequentially for now.  Change so it draws randomly
function get_train_test_data(data_dir, num_train, num_test)
local segdir = path.join(path.abspath(data_dir), 'PKLotSegmented')
  local trainfiles = {}
  local testfiles = {}
  local n = 1
  
  for lotdir in paths.iterdirs(segdir) do
    for weatherdir in paths.iterdirs(path.join(segdir, lotdir)) do
      for datedir in paths.iterdirs(path.join(segdir, lotdir, weatherdir)) do
        for vacancydir in paths.iterdirs(path.join(segdir, lotdir, weatherdir, datedir)) do
          for file in paths.iterfiles(path.join(segdir, lotdir, weatherdir, datedir, vacancydir)) do
            local filepath = path.join(segdir, lotdir, weatherdir, datedir, vacancydir, file)
            if n <= num_train then
              table.insert(trainfiles, filepath)
            elseif n <= num_train + num_test then
              table.insert(testfiles, filepath)
            else
              break
            end
            n = n+1
          end
        end
      end    
    end
  end
  
  return trainfiles, testfiles
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

trainfiles, testfiles = get_train_test_data(data_dir, 100, 100)
print('Training files: ' .. #trainfiles)
print('Test files: ' .. #testfiles)

-- for i = 1, 10 do
--   print(trainfiles[i])
-- end
-- for i = 1, 10 do
--   print(testfiles[i])
-- end