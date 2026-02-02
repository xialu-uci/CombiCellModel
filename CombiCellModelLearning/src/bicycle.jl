
using CombiCellModelLearning
using ComponentArrays # i feel like I shouldn't need this in here...
using Optimization
using OptimizationBBO
# using OptimizationBBO

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
    O2max=110.0
)


# basically copied over from fw() in define_modelCombi_classical.jl
function true_fw(x::Vector{Float64}, kD::Vector{Float64}, params)
    # unpack params
    fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max = params

    O1 = Float64[]
    O2 = Float64[]

    for (xi, kDi) in zip(x, kD)
        CT = (1-fI) * (alpha * xi + tT + g1 * kDi / k_on_2d - sqrt((alpha * xi + tT + g1 * kDi / k_on_2d)^2 - 4 * alpha * xi * tT)) / 2
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

fakeData = Dict(
    "x" => x_for_sim,
    "KD" => kD_for_sim,
    # stdevs = stdevs_for_sim,
    "O1_00" => O1_sim_data,
    "O2_00" => O2_sim_data
)

# now let's make a classical model and try to fit parameters to the simulated data
# differential evolution


model = CombiCellModelLearning.make_ModelCombiClassic()
p_repr_ig = deepcopy(model.params_repr_ig)
# learning problem
learning_problem = CombiCellModelLearning.LearningProblem(
    data =fakeData,
    model= model,
    p_repr_lb=CombiCellModelLearning.represent(model.p_derepresented_lowerbounds, model),
    p_repr_ub=CombiCellModelLearning.represent(model.p_derepresented_upperbounds, model),
    mask = trues(length(x_for_sim)), # no mask for now
    loss_strategy="vanilla")

# for bbo, use solve(prob, algo, maxiters, callback)

# prob is an optimizationproblem
# OptimizationProblem(objective_function, initial_guess, p = either constant prams or fixed used in objective function?? = [1,100.0]???, lb, ub)

# for objective function, need to define function that takes in parameter array and returns loss (pass get_loss with reconstructed params)
function obj_func(x, p)
    p_repr = CombiCellModelLearning.reconstruct_learning_params_from_array(x, p_repr_ig, model) # this is where params are updated # the trick is the x is the actual params we want. 
    # only pass through p_repr_ig for the keys
    return CombiCellModelLearning.get_loss(p_repr; learning_problem=learning_problem)
end

# initial guess params array
classical_params_array = collect(values(copy(p_repr_ig)))

# p 

p = [1.0, 100.0] # not used in obj_func, honestly I think i could delete this but will leave in for now and see if it changes anything later

# algo is the optimization algorithm (here bbo)
# maxiters is maximum iterations
maxiters = 30000 # reduced for testing
# callback is a function called at each iteration, s.t. optimzation stops if it returns true
config = CallbackConfig() # just stores info for callback function in fields
callback, loss_history = CombiCellModelLearning.create_bbo_callback_with_early_termination(
    config, maxiters)

# create optimization problem
prob = Optimization.OptimizationProblem(
    obj_func,
    classical_params_array,
    p;
    lb= learning_problem.p_repr_lb,
    ub=learning_problem.p_repr_ub)

# solve optimization problem
sol = solve(prob, BBO_adaptive_de_rand_1_bin(); callback=callback, maxiters=maxiters)
final_params = CombiCellModelLearning.reconstruct_learning_params_from_array(sol.minimizer, p_repr_ig, learning_problem.model)

# plot loss history
using Plots 
plot(loss_history, xlabel="Iteration", ylabel="Loss", title="BBO Loss History") # to do check plot
# save plot
savefig("bbo_loss_history.png")
# print final params
println("Final parameters found by BBO:")
for (name, value) in zip(keys(final_params.p_classical), values(final_params.p_classical))
    println("$name: $value")
end


