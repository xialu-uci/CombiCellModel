function sim_data(x, kD, stdevs, params, true_fw)
    # x is vector of input concentrations
    # kD is vector of dissociation constants
    # stdevs is vector of standard deviations for fake experimental outputs
    # model_fw is function that takes (x, kD, params) and returns predicted outputs
    num_points = length(x)
    # sim_O1_00, sim_O2_00 = true_fw(x, kD, params) # was used for no accessory
    O1_00, O2_00, O1_10, O2_10, O1_01, O2_01, O1_11, O2_11 = true_fw(x, kD, params)
    outputs = [O1_00, O2_00, O1_10, O2_10, O1_01, O2_01, O1_11, O2_11]

    # Add noise to simulate experimental data
    # noise1 = randn(num_points) .* stdevs
    # sim_O1_00 .= sim_O1_00 .+ noise1
    # noise2 = randn(num_points) .* stdevs
    # sim_O2_00 .= sim_O2_00 .+ noise2

    # add noise to all
    for output in outputs
        noise = randn(num_points).*stdevs
        output += noise
    end
    

    fakeData = Dict(
        "x" => x,
        "KD" => kD,
        "O1_00" => O1_00,
        "O2_00" => O2_00,
        "O1_10" => O1_10,
        "O2_10" => O2_10,
        "O1_01" => O1_01,
        "O2_01" => O2_01,
        "O1_11" => O1_11,
        "O2_11" => O2_11
    )

    return fakeData
end


