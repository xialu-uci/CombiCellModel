
using CombiCellModelLearning
using ComponentArrays # i feel like I shouldn't need this in here...
include("sim_data.jl")

x_for_sim = [0.1, 0.02, 0.004, 0.0008, 0.00016, 0.000032, 0.0000064, 0.00000128, 0.000000256, 5.12e-8, 1.024e-8, 0.00]
kD_for_sim = [8.0 for _ in x_for_sim]
stdevs_for_sim = [0.005 for _ in x_for_sim]

# function fw_sim(x::Vector{Float64}, kD::Vector{Float64}, p_derepresented)
#     model = make_ModelCombiClassic()
#     O1_00 = zeros(length(x))
#     O2_00 = zeros(length(x))
#     for i in 1:length(x)
#         O1_00[i], O2_00[i] = fw(x[i], kD[i], p_derepresented, model)
#     end
#     return (O1_00, O2_00)
# end

params_for_sim = ComponentArray(
    fI=0.5,
    alpha=1e6,
    tT=1e3,
    g1=0.5,
    k_on_2d=10.0,
    kP=0.5,
    nKP=2.0,
    lambdaX=0.01,
    nC=2.0,
    XO1=0.5,
    O1max=0.5,
    O2max=100.0
)


# basically copied over from fw() in define_modelCombi_classical.jl
function true_fw(x::Vector{Float64}, kD::Vector{Float64}, params)
    # unpack params
    fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max = params

    O1 = Float64[]
    O2 = Float64[]

    for (xi, kDi) in zip(x, kD)
        CT = (alpha * xi + tT + g1 * kDi / k_on_2d - sqrt((alpha * xi + tT + g1 * kDi / k_on_2d)^2 - 4 * alpha * xi * tT)) / 2
        CN = (1 / (1 + g1 * kDi / kP))^nKP * CT
        X = CN^nC / (lambdaX^nC + CN^nC)

        O1_val = XO1 / (XO1 + X)
        O2_val = X

        O1i = O1max * O1_val 
        O2i = O2max * O2_val

        push!(O1, O1i)
        push!(O2, O2i)
    end
    return (O1, O2)
end

O1_sim_data, O2_sim_data = sim_data(x_for_sim, kD_for_sim, stdevs_for_sim, params_for_sim, true_fw)
