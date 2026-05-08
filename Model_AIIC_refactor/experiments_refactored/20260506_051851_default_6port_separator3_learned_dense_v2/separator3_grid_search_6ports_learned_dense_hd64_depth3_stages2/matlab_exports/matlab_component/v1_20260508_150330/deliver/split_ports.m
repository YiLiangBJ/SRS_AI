function portData = split_ports(outputData, portIndex)
%SPLIT_PORTS Return one port slice from N x num_ports x (2*seq_len) output.
validateattributes(outputData, {'numeric'}, {'3d'});
validateattributes(portIndex, {'numeric'}, {'scalar', 'integer', 'positive'});
if portIndex > size(outputData, 2)
    error('split_ports:BadPortIndex', 'portIndex=%d exceeds available ports=%d.', portIndex, size(outputData, 2));
end
portData = squeeze(outputData(:, portIndex, :));
if size(outputData, 1) == 1
    portData = reshape(portData, 1, []);
end
end
