function state = init_model(componentDir)
%INIT_MODEL One-time initialization for deployed Matlab inference.
if nargin < 1 || isempty(componentDir)
    componentDir = fileparts(mfilename('fullpath'));
end
state = load_srs_ai_matlab_component(componentDir);
state.component_dir = string(componentDir);
state.model_name = "sep3_hd64_d3_s2";
state.seq_len = double(state.manifest.model_spec.seq_len);
state.input_width = state.seq_len * 2;
state.num_ports = double(state.manifest.model_spec.num_ports);
state.output_width = state.input_width;
end
