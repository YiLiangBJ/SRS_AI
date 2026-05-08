function ports = split_ports(outputData)
%SPLIT_PORTS Demo-local wrapper so scripts under demo/ run directly.
componentDir = fileparts(fileparts(mfilename('fullpath')));
addpath(componentDir);
ports = feval('split_ports', outputData);
end
