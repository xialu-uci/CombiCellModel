function sim_data(x, kD, stdevs, params, model_fw)
    # x is vector of input concentrations
    # kD is vector of dissociation constants
    # stdevs is vector of standard deviations for fake experimental outputs
    # model_fw is function that takes (x, kD, params) and returns predicted outputs
    num_points = length(x)
    sim_O1_00 = zeros(num_points)
    sim_O2_00 = zeros(num_points)
    for i in 1:num_points
        sim_O1_00[i], sim_O2_00[i] = model_fw(x[i], kD[i], params)
        # Add noise
        sim_O1_00[i] += randn() * stdevs[i] # randn no inputs is standard normal
        sim_O2_00[i] += randn() * stdevs[i] # randn no inputs is standard normal
    end
    return (sim_O1_00, sim_O2_00)
end


