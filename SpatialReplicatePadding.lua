local SpatialReplicatePadding, parent = torch.class('nn.SpatialReplicatePadding', 'nn.Module')

-- Pad feature maps by replicating nearest values
function SpatialReplicatePadding.padRep(input, padW, padH)
    -- Get input size
    local z = input:size(1) -- channels
    local r = input:size(2) -- rows
    local c = input:size(3) -- cols
    -- Initialize padded output
    local out = torch.zeros(z, r+2*padH, c+2*padW)
    -- Copy original image
    out[{{}, {padH+1, r+padH}, {padW+1, c+padW}}]:copy(input)
    -- Pad left (no corners)
    local dst = out[{{},{padH+1,padH+r},{1,padW}}]
    dst:copy(out[{{},{padH+1,padH+r},{padW+1,padW+1}}]:expandAs(dst))
    -- Pad right (no corners)
    dst = out[{{},{padH+1,padH+r},{padW+c+1,2*padW+c}}]
    dst:copy(out[{{},{padH+1,padH+r},{padW+c,padW+c}}]:expandAs(dst))
    -- Pad top (with corners)
    dst = out[{{},{1,padH},{1,2*padW+c}}]
    dst:copy(out[{{},{padH+1,padH+1},{1,2*padW+c}}]:expandAs(dst))
    -- Pad bottom (with corners)
    dst = out[{{},{padH+r+1,2*padH+r},{1,2*padW+c}}]
    dst:copy(out[{{},{padH+r,padH+r},{1,2*padW+c}}]:expandAs(dst))
    -- Return
    return out
end

function SpatialReplicatePadding:__init(padH, padW)
    parent.__init(self)
    -- Check pad value
    if padH < 1 or padW < 1 then
        error('nn.SpatialReplicatePadding: pad values must be positive')
    end
    self.padH = padH
    self.padW = padW
end

function SpatialReplicatePadding:updateOutput(input)
    -- Compute padded output
    local output = self.padRep(input, self.padH, self.padW)
    -- Set output
    self.output:resizeAs(output):copy(output)
    return self.output
end

-- No idea if this is right
function SpatialReplicatePadding:updateGradInput(input, gradOutput)
    -- Crop grad output
    local cropped_gradOutput = gradOutput[{{}, {1+self.padH,1+self.padH+input:size(2)}, {1+self.padW,1+self.padW+input:size(3)}}]
    self.gradInput:resizeAs(input):copy(cropped_gradOutput)
    return self.gradInput
end
