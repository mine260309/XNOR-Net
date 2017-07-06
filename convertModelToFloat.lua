require 'torch'
require 'cutorch'
--require 'paths'
--require 'xlua'
require 'optim'
require 'nn'
require 'newLayers'

require 'cunn'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:text('Convert a model from cuda to float for CPU use')
cmd:text('Options:')
cmd:option('-from', 'model_from', 'The model to convert from')
cmd:option('-to', 'model_to', 'The model to convert to')
local opt = cmd:parse(arg or {})
print("Converting model from \"" .. opt['from'] .. "\" to \"" .. opt['to'] .. "\"")

model = torch.load(opt['from'])
--print(model)
model:float()
cudnn.convert(model, nn)
torch.save(opt['to'], model)
print("Done")
