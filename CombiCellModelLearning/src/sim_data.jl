using JLD2

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


x_for_sim = [0.1, 0.02, 0.004, 0.0008, 0.00016, 0.000032, 0.0000064, 0.00000128, 0.000000256, 5.12e-8, 1.024e-8, 0.00, 0.1, 0.02, 0.004, 0.0008, 0.00016, 0.000032, 0.0000064, 0.00000128, 0.000000256, 5.12e-8, 1.024e-8, 0.00, 0.1, 0.02, 0.004, 0.0008, 0.00016, 0.000032, 0.0000064, 0.00000128, 0.000000256, 5.12e-8, 1.024e-8, 0.00]

# Each value repeated for one-third of the length of x_for_sim
segment_length = div(length(x_for_sim), 3)
kD_for_sim = vcat(
    fill(8.0, segment_length),
    fill(50.0, segment_length),
    fill(200.0, segment_length)
)

stdevs_for_sim = fill(0.005, length(x_for_sim))


params_for_sim = ComponentArray(
    fI=0.6,
    alpha=2e6,
    tT=500.0,
    g1=0.7,
    k_on_2d=15.0,
    kP=0.4,
    nKP=1.8,
    lambdaX=0.05,
    nC=2.1,
    XO1=0.4,
    O1max=0.95,
    O2max=110.0,
    extraCD2 = 0.99,
    extraPD1 = 65
)


# basically copied over from fw() in define_modelCombi_classical.jl

function true_fw(x::Vector{Float64}, kD::Vector{Float64}, params)
    # unpack params
    fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max, extraCD2, extraPD1 = params
    # hypothesis = CD2 affects O2max, PD1 affects O1max
    params_00 = ComponentArray(
        fI=fI,
        alpha=alpha,
        tT=tT,
        g1=g1,
        k_on_2d=k_on_2d,
        kP=kP,
        nKP=nKP,
        lambdaX=lambdaX,
        nC=nC,
        XO1=XO1,
        O1max=O1max,
        O2max=O2max,
    )

    O1_00, O2_00 = true_fw_inside(x, kD, params_00)

    params_10 = ComponentArray(
        fI=fI,
        alpha=alpha,
        tT=tT,
        g1=g1,
        k_on_2d=k_on_2d,
        kP=kP,
        nKP=nKP,
        lambdaX=lambdaX,
        nC=nC,
        XO1=XO1,
        O1max=extraCD2,
        O2max=O2max,
    )

    O1_10, O2_10 = true_fw_inside(x, kD, params_10)


    params_01 = ComponentArray(
        fI=fI,
        alpha=alpha,
        tT=tT,
        g1=g1,
        k_on_2d=k_on_2d,
        kP=kP,
        nKP=nKP,
        lambdaX=lambdaX,
        nC=nC,
        XO1=XO1,
        O1max=O1max,
        O2max=extraPD1,
    )

    O1_01, O2_01 = true_fw_inside(x, kD, params_01)

    params_11 = ComponentArray(
        fI=fI,
        alpha=alpha,
        tT=tT,
        g1=g1,
        k_on_2d=k_on_2d,
        kP=kP,
        nKP=nKP,
        lambdaX=lambdaX,
        nC=nC,
        XO1=XO1,
        O1_max=extraCD2,
        O2_max=extraPD1,
    )
    
    O1_11, O2_11 = true_fw_inside(x, kD, params_11)

    return O1_00, O2_00,O1_10, O2_10, O1_01, O2_01, O1_11, O2_11
    
    
end
function true_fw_inside(x::Vector{Float64}, kD::Vector{Float64}, params)
    # unpack params
    fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max = params

    O1 = Float64[]
    O2 = Float64[]

    for (xi, kDi) in zip(x, kD)
        CT = (1-fI) * (alpha * xi + tT + g1 * kDi / k_on_2d - sqrt((alpha * xi + tT + g1 * kDi / k_on_2d)^2 - 4 * alpha * xi * tT)) / 2
        CN = (1 / (1 + g1 * kDi / kP))^nKP * CT
        X = CN^nC / (lambdaX^nC + CN^nC)

        O1_val = X / (XO1 + X)
        O2_val = X

        O1i = O1max * O1_val 
        O2i = O2max * O2_val

        push!(O1, O1i)
        push!(O2, O2i)
    end
    return (O1, O2)
end

fakeData = sim_data(x_for_sim, kD_for_sim, stdevs_for_sim, params_for_sim, true_fw)

base_path = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/"

@save joinpath(base_path, "data/fakeData.jld2") fakeData



