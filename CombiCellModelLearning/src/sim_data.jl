function sim_data(x, kD, stdevs, params, true_fw)
    # x is vector of input concentrations
    # kD is vector of dissociation constants
    # stdevs is vector of standard deviations for fake experimental outputs
    # model_fw is function that takes (x, kD, params) and returns predicted outputs
    num_points = length(x)
    sim_O1_00, sim_O2_00 = true_fw(x, kD, params)
    # Add noise to simulate experimental data
    noise1 = randn(num_points) .* stdevs
    sim_O1_00 .= sim_O1_00 .+ noise1
    noise2 = randn(num_points) .* stdevs
    sim_O2_00 .= sim_O2_00 .+ noise2

    (sim_O1_00, sim_O2_00)
end


