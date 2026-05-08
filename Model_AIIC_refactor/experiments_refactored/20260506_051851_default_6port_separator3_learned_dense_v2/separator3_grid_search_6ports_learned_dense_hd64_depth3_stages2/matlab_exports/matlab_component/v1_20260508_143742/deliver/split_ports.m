function ports = split_ports(outputData)
%SPLIT_PORTS Convert N x 6 x 24 output into a 1x6 cell array of N x 24 slices.
validateattributes(outputData, {'numeric'}, {'3d'});
numPorts = size(outputData, 2);
ports = cell(1, numPorts);
for portIdx = 1:numPorts
    ports{portIdx} = squeeze(outputData(:, portIdx, :));
    if size(outputData, 1) == 1
        ports{portIdx} = reshape(ports{portIdx}, 1, []);
    end
end
end
