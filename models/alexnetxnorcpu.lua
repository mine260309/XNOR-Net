function createModel()
require 'nn'
   local function activation()
      local C= nn.Sequential()
      C:add(nn.BinActiveZ())
      return C
   end

   local function MaxPooling(kW, kH, dW, dH, padW, padH)
    return nn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
   end

   local function BinConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
         local C= nn.Sequential()
          C:add(nn.SpatialBatchNormalization(nInputPlane,1e-4,false))
          C:add(activation())
          C:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
   		 return C
   end

    local function BinMaxConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH,mW,mH)
         local C= nn.Sequential()
          C:add(nn.SpatialBatchNormalization(nInputPlane,1e-4,false))
          C:add(activation())
          C:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
          C:add(MaxPooling(3,3,2,2))
       return C
   end

local features = nn.Sequential()

   features:add(nn.SpatialConvolution(3,96,11,11,4,4,2,2))
   features:add(nn.SpatialBatchNormalization(96,1e-5,false))
   features:add(nn.ReLU(true))
   features:add(MaxPooling(3,3,2,2))

   features:add(BinMaxConvolution(96,256,5,5,1,1,2,2))
   features:add(BinConvolution(256,384,3,3,1,1,1,1))
   features:add(BinConvolution(384,384,3,3,1,1,1,1))
   features:add(BinMaxConvolution(384,256,3,3,1,1,1,1))
   features:add(BinConvolution(256,4096,6,6))
   features:add(BinConvolution(4096,4096,1,1))

   features:add(nn.SpatialBatchNormalization(4096,1e-3,false))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(4096, nClasses,1,1))

   features:add(nn.View(nClasses))
   features:add(nn.LogSoftMax())

   local model = features
   return model
end
