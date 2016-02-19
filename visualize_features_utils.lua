-- Index to subscript
-- Didn't know about the existence of :nonzero()
function ind2sub(x, size)
    -- Check x is vector
    -- TODO
    -- Get number of dimensions
    local ndims = size:size()
    -- Check input is float tensor
    if type(x) == "number" then
        x = torch.Tensor(1):fill(x)
    else
        x = x:float()
    end
    -- Get input length
    local n = x:nElement()
    -- Initialize subscript
    local x_sub = torch.zeros(ndims, n)
    for i=1,ndims do
        local d = 1
        if i < ndims then
            for j=i+1,ndims do d = d*size[j] ; end
        end
        x_sub[i] = torch.ceil(torch.div(x,d))
        local delta_x = torch.add(x_sub[i], -1):mul(d)
        x:add(-delta_x)
    end
    return x_sub:long():t()
end

-- Unique
-- Gets tensor, returns table
function unique(input)
    local b = {}
    for i=1,input:nElement() do
        b[input[i]] = true
    end
    local out = {}
    for i in pairs(b) do
        table.insert(out,i)
    end
    return out
end

-- Pad feature maps by replicating nearest values
function padRep(input, pad)
    -- Get input size
    local z = input:size(1) -- channels
    local r = input:size(2) -- rows
    local c = input:size(3) -- cols
    -- Initialize padded output
    local out = torch.zeros(z, r+2*pad, c+2*pad):long()
    -- Copy original image
    out[{{}, {pad+1, r+pad}, {pad+1, c+pad}}]:copy(input)
    -- Pad left (no corners)
    local dst = out[{{},{pad+1,pad+r},{1,pad}}]
    dst:copy(out[{{},{pad+1,pad+r},{pad+1,pad+1}}]:expandAs(dst))
    -- Pad right (no corners)
    dst = out[{{},{pad+1,pad+r},{pad+c+1,pad+c+pad}}]
    dst:copy(out[{{},{pad+1,pad+r},{pad+c,pad+c}}]:expandAs(dst))
    -- Pad top (with corners)
    dst = out[{{},{1,pad},{1,2*pad+c}}]
    dst:copy(out[{{},{pad+1,pad+1},{1,2*pad+c}}]:expandAs(dst))
    -- Pad bottom (with corners)
    dst = out[{{},{pad+r+1,2*pad+r},{1,2*pad+c}}]
    dst:copy(out[{{},{pad+r,pad+r},{1,2*pad+c}}]:expandAs(dst))
    -- Return
    return out
end

-- Select activation neurons
function getActiveNeurons(conv_net, target_layer, target_map)
    -- Get feature maps
    local rec_input = conv_net.modules[target_layer].output:clone()
    -- Sort by activation
    activation_values = rec_input:reshape(rec_input:nElement())
    activation_values, activation_idx = torch.sort(activation_values, 1, true)
    -- Convert to map coordinates
    activation_coords = ind2sub(activation_idx, rec_input:size())
    -- If required, filter by map
    if target_map ~= nil then
        -- Map coordinate is first one
        selected_rows = activation_coords[{{},1}]:eq(target_map):nonzero():squeeze()
        -- Filter
        activation_values = activation_values:index(1, selected_rows)
        activation_coords = activation_coords:index(1, selected_rows)
    end
    -- Return
    return activation_values:clone(), activation_coords:clone()
end

-- Filter neurons by removing near-duplicate activations
function filterNeurons(active_values, active_coords, limit, margin, allow_map_overlap)
    -- Initialize output
    local new_active_values = torch.Tensor(limit)
    local new_active_coords = torch.Tensor(limit, 3)
    local new_size = 0
    local curr = 1
    -- Add items
    while new_size < limit do
        -- Check current element exists
        if curr > active_values:nElement() then
            print("filterNeurons: Warning: could not find " .. limit .. " neurons, returning " .. new_size)
            break
        end
        -- Get current neuron
        local value = active_values[curr]
        local coords = active_coords[curr]
        -- Check if it is too close to neurons already added
        local can_add = true
        if new_size > 0 then
            can_add = false
            -- Get view of current new neuron list
            local curr_new_active_values = new_active_values[{{1,new_size}}]
            local curr_new_active_coords = new_active_coords[{{1,new_size},{}}]
            -- Get neurons in the same map
            -- This should be rewritten, I'm hacking the previous implementation
            local same_map_idx = torch.ones(curr_new_active_coords:size(1)):byte()
            if allow_map_overlap then
                same_map_idx = curr_new_active_coords:t()[1]:eq(coords[1])
            end
            same_map_idx = torch.linspace(1, same_map_idx:nElement(), same_map_idx:nElement())[same_map_idx]:long()
            -- Check if there are neurons in the same map
            if same_map_idx:sum() > 0 then
                curr_new_active_values = curr_new_active_values:index(1, same_map_idx)
                curr_new_active_coords = curr_new_active_coords:index(1, same_map_idx):clone()
                -- Subtract neuron coords
                --print(active_coords:isContiguous())
                curr_new_active_coords:add(-coords:view(1,3):expandAs(curr_new_active_coords):float()):abs()
                -- Check any element is beyond margin
                if torch.min(curr_new_active_coords[{{}, {2}}]) > margin or torch.min(curr_new_active_coords[{{}, {3}}]) > margin then
                    -- No near neuron found, can save
                    can_add = true
                end
            else
                -- First neuron for this map, let's add it
                can_add = true
            end
        end
        -- Add to list
        if can_add then
            -- Increase new size
            new_size = new_size + 1
            new_active_values[new_size] = value
            new_active_coords[new_size]:copy(coords)
        end
        -- Increase current
        curr = curr + 1
    end  
    -- Remove excess elements
    new_active_values = new_active_values[{{1,new_size}}]
    new_active_coords = new_active_coords[{{1,new_size},{}}]
    collectgarbage()
    -- Return values
    return new_active_values, new_active_coords
end

-- Generate reconstruction input
-- Must be called with the same target_layer and target_map values as used for getActiveNeurons
function getReconstructionInput(activation_values, activation_coords, conv_net, target_layer, target_map)
    -- Define output table
    local rec_inputs = {}
    -- Get output sizes
    local rec_input_size = conv_net.modules[target_layer].output:size()
    -- Process each neuron
    for i=1,activation_values:nElement() do
        -- Get neuron info
        local coords = torch.totable(activation_coords[i])
        local value = activation_values[i]
        -- Initialize new maps
        local rec_input = torch.zeros(rec_input_size)
        rec_input[coords] = value
        -- Add to list
        table.insert(rec_inputs, rec_input)
    end
    -- Return result
    return rec_inputs
end

function buildReconstructionNet(conv_net, target_layer)
    -- Initialize deconv net
    local deconv_net = nn.Sequential()
    -- Process layers
    for i=1,target_layer do
        -- Get module
        local m = conv_net.modules[i]
        -- Check convolutional layer
        if torch.typename(m) == "nn.SpatialConvolutionMM" then
            -- Configure deconvolutional layer
            local deconv = m:clone()
            deconv.nInputPlane = m.nOutputPlane
            deconv.nOutputPlane = m.nInputPlane
            deconv.bias:fill(0)
            deconv.weight = deconv.weight:reshape(m.nOutputPlane, m.nInputPlane, m.kW, m.kH)
            deconv.weight = image.flip(deconv.weight, 3)
            deconv.weight = image.flip(deconv.weight, 4)
            deconv.weight = deconv.weight:transpose(1,2)
            deconv.weight = deconv.weight:reshape(deconv.nOutputPlane, deconv.nInputPlane*deconv.kW*deconv.kH)
            -- Add deconvolutional layer to deconv net
            deconv_net:insert(deconv, 1)
            -- Add replicate-padding layer to deconv net
            deconv_net:insert(nn.SpatialReplicatePadding((m.kW-1)/2, (m.kH-1)/2), 1)
        elseif torch.typename(m) == "nn.ReLU" then
            -- Add ReLU to deconv net
            deconv_net:insert(nn.ReLU(), 1)
        elseif torch.typename(m) == "nn.SpatialMaxPooling" then
            -- Add unpooler to deconv net
            deconv_net:insert(nn.SpatialMaxUnpooling(m), 1)
        end
    end
    -- Return deconv net
    return deconv_net
end

-- Localize main activity in map
-- Input should be CxMxN
-- Output is binary MxN
function localizeActivity(rec_output, opts)
    -- Check opts
    if opts == nil then
        opts = {}
    end
    -- Compute absolute value
    local activity_map = torch.abs(rec_output)
    -- Flatten activity mask
    activity_map = torch.mean(activity_map, 1)
    -- Threshold activity map
    local activity_thr = opts.activity_thr or 0.0001
    local activity_mask = activity_map:ge(activity_thr):squeeze()
    -- Return mask
    return activity_mask
end

-- Get coordinates of top-left point and bottom-right point of binary mask
-- Input mask should be M1xN1
function getMaskCoords(activity_mask)
    -- Compute crop
    crop_min_x = nil
    crop_min_y = nil
    crop_max_x = nil
    crop_max_y = nil
    for i=1,activity_mask:size(1) do
        for j=1,activity_mask:size(2) do
            if activity_mask[{i,j}] == 1 and (crop_min_x == nil or j < crop_min_x) then crop_min_x = j end
            if activity_mask[{i,j}] == 1 and (crop_min_y == nil or i < crop_min_y) then crop_min_y = i end
            if activity_mask[{i,j}] == 1 and (crop_max_x == nil or j > crop_max_x) then crop_max_x = j end
            if activity_mask[{i,j}] == 1 and (crop_max_y == nil or i > crop_max_y) then crop_max_y = i end
        end
    end
    -- Return coordinates
    return {crop_min_x, crop_min_y, crop_max_x, crop_max_y}
end

-- Draw rectangle
function drawRectangle(img, p1, p2, color)
    local p1x, p1y = unpack(p1)
    local p2x, p2y = unpack(p2)
    -- top line
    local top_line = img[{{}, {p1y}, {p1x,p2x}}]
    top_line:copy(color:view(3,1,1):expandAs(top_line))
    -- bottom line
    local bottom_line = img[{{}, {p2y}, {p1x,p2x}}]
    bottom_line:copy(color:view(3,1,1):expandAs(bottom_line))
    -- left line
    local left_line = img[{{}, {p1y,p2y}, {p1x}}]
    left_line:copy(color:view(3,1,1):expandAs(left_line))
    -- right line
    local right_line = img[{{}, {p1y,p2y}, {p2x}}]
    right_line:copy(color:view(3,1,1):expandAs(right_line))
end
