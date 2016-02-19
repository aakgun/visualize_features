local OverFeatFactory = {}

function OverFeatFactory.create(params)
    -- Get references to nn functions
    local SpatialConvolution = nn.SpatialConvolution
    local SpatialConvolutionMM = nn.SpatialConvolutionMM
    local SpatialMaxPooling = nn.SpatialMaxPooling
    -- Define CNN model
    local conv_net = nn.Sequential() 
    if params.type == "big" then
        -- Define network
        conv_net:add(SpatialConvolution(3, 96, 7, 7, 2, 2))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialMaxPooling(3, 3, 3, 3))
        conv_net:add(SpatialConvolutionMM(96, 256, 7, 7, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialMaxPooling(2, 2, 2, 2))
        conv_net:add(SpatialConvolutionMM(256, 512, 3, 3, 1, 1, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialConvolutionMM(512, 512, 3, 3, 1, 1, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialConvolutionMM(512, 1024, 3, 3, 1, 1, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialConvolutionMM(1024, 1024, 3, 3, 1, 1, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialMaxPooling(3, 3, 3, 3))
        conv_net:add(SpatialConvolutionMM(1024, 4096, 5, 5, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialConvolutionMM(4096, 4096, 1, 1, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialConvolutionMM(4096, 1000, 1, 1, 1, 1))
    elseif params.type == "small" then
        -- Define network
        conv_net:add(SpatialConvolutionMM(3, 96, 11, 11, 4, 4))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialMaxPooling(2, 2, 2, 2))
        conv_net:add(SpatialConvolutionMM(96, 256, 5, 5, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialMaxPooling(2, 2, 2, 2))
        conv_net:add(SpatialConvolutionMM(256, 512, 3, 3, 1, 1, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialConvolutionMM(512, 1024, 3, 3, 1, 1, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialConvolutionMM(1024, 1024, 3, 3, 1, 1, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialMaxPooling(2, 2, 2, 2))
        conv_net:add(SpatialConvolutionMM(1024, 3072, 6, 6, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialConvolutionMM(3072, 4096, 1, 1, 1, 1))
        conv_net:add(nn.ReLU(true))
        conv_net:add(SpatialConvolutionMM(4096, 1000, 1, 1, 1, 1))
    end
    local loss_net = nn.Sequential()
    loss_net:add(nn.View(1000))
    loss_net:add(nn.LogSoftMax())
    -- Check load pretrained weights
    if params.pretrained then
        -- Require
        local ParamBank = require 'ParamBank'
        -- Check weights file exists
        local weights_exist = io.open(params.weights_path)
        if weights_exist == nil then
            error("Missing weights file")
        end
        io.close(weights_exist)
        -- Get modules
        local m = conv_net.modules
        -- Open weights file
        ParamBank:init(params.weights_path)
        -- Fill weights
        local offset  = 0
        if params.type == "big" then
            ParamBank:read(        0, {96,3,7,7},      m[offset+1].weight)
            ParamBank:read(    14112, {96},            m[offset+1].bias)
            ParamBank:read(    14208, {256,96,7,7},    m[offset+4].weight)
            ParamBank:read(  1218432, {256},           m[offset+4].bias)
            ParamBank:read(  1218688, {512,256,3,3},   m[offset+7].weight)
            ParamBank:read(  2398336, {512},           m[offset+7].bias)
            ParamBank:read(  2398848, {512,512,3,3},   m[offset+9].weight)
            ParamBank:read(  4758144, {512},           m[offset+9].bias)
            ParamBank:read(  4758656, {1024,512,3,3},  m[offset+11].weight)
            ParamBank:read(  9477248, {1024},          m[offset+11].bias)
            ParamBank:read(  9478272, {1024,1024,3,3}, m[offset+13].weight)
            ParamBank:read( 18915456, {1024},          m[offset+13].bias)
            ParamBank:read( 18916480, {4096,1024,5,5}, m[offset+16].weight)
            ParamBank:read(123774080, {4096},          m[offset+16].bias)
            ParamBank:read(123778176, {4096,4096,1,1}, m[offset+18].weight)
            ParamBank:read(140555392, {4096},          m[offset+18].bias)
            ParamBank:read(140559488, {1000,4096,1,1}, m[offset+20].weight)
            ParamBank:read(144655488, {1000},          m[offset+20].bias)
        elseif params.type == "small" then
            ParamBank:read(        0, {96,3,11,11},    m[offset+1].weight)
            ParamBank:read(    34848, {96},            m[offset+1].bias)
            ParamBank:read(    34944, {256,96,5,5},    m[offset+4].weight)
            ParamBank:read(   649344, {256},           m[offset+4].bias)
            ParamBank:read(   649600, {512,256,3,3},   m[offset+7].weight)
            ParamBank:read(  1829248, {512},           m[offset+7].bias)
            ParamBank:read(  1829760, {1024,512,3,3},  m[offset+9].weight)
            ParamBank:read(  6548352, {1024},          m[offset+9].bias)
            ParamBank:read(  6549376, {1024,1024,3,3}, m[offset+11].weight)
            ParamBank:read( 15986560, {1024},          m[offset+11].bias)
            ParamBank:read( 15987584, {3072,1024,6,6}, m[offset+14].weight)
            ParamBank:read(129233792, {3072},          m[offset+14].bias)
            ParamBank:read(129236864, {4096,3072,1,1}, m[offset+16].weight)
            ParamBank:read(141819776, {4096},          m[offset+16].bias)
            ParamBank:read(141823872, {1000,4096,1,1}, m[offset+18].weight)
            ParamBank:read(145919872, {1000},          m[offset+18].bias)
        end
        -- Close weights file
        ParamBank:close()
    end
    -- Return model
    return conv_net, loss_net
end

return OverFeatFactory
