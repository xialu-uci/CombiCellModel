
using CombiCellModelLearning
using ComponentArrays # i feel like I shouldn't need this in here...
using Optimization
using OptimizationBBO
using Statistics
using JLD2
# using Makie
# using CairoMakie



loaddir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/data/" # modify for hpc
@load joinpath(loaddir, "fakeData.jld2") fakeData
# @load joinpath(loaddir, "CombiCell_data.jld2") data

# now let's make a classical model and try to fit parameters to the simulated data
# differential evolution


model = CombiCellModelLearning.make_ModelCombiClassic()

p_repr_ig = deepcopy(model.params_repr_ig)
# learning problem
learning_problem = CombiCellModelLearning.LearningProblem(
     data =fakeData, # or data (real)
     model= model,
     p_repr_lb=CombiCellModelLearning.represent(model.p_derepresented_lowerbounds, model),
     p_repr_ub=CombiCellModelLearning.represent(model.p_derepresented_upperbounds, model),
     mask = trues(length(x_for_sim)), # no mask for now
     loss_strategy="normalized")

# for bbo, use solve(prob, algo, maxiters, callback)

# prob is an optimizationproblem
# OptimizationProblem(objective_function, initial_guess, p = either constant prams or fixed used in objective function?? = [1,100.0]???, lb, ub)

# # TODO: move below functions to xialuDiffEvol
# # for objective function, need to define function that takes in parameter array and returns loss (pass get_loss with reconstructed params)
# function obj_func(x, p)
#     p_repr = CombiCellModelLearning.reconstruct_learning_params_from_array(x, p_repr_ig, model) # this is where params are updated # the trick is the x is the actual params we want. 
#     # only pass through p_repr_ig for the keys
#     return CombiCellModelLearning.get_loss(p_repr; learning_problem=learning_problem)
# end

# # let's make this whole section a function bbo_learn(learning_problem, p_repr_ig)
# function bbo_learn(learning_problem, p_repr_ig)
#     # initial guess params array
#     classical_params_array = collect(values(copy(p_repr_ig)))

#     # p
#     p = [1.0, 100.0] # not used in obj_func, honestly I think i could delete this but will leave in for now and see if it changes anything later

#     # algo is the optimization algorithm (here bbo)
#     # maxiters is maximum iterations
#     maxiters = 30000 # reduced for testing
#     # callback is a function called at each iteration, s.t. optimzation stops if it returns true
#     config = CallbackConfig() # just stores info for callback function in fields
#     callback, loss_history = CombiCellModelLearning.create_bbo_callback_with_early_termination(
#     config, maxiters)

#     # create optimization problem
#     prob = Optimization.OptimizationProblem(
#         obj_func,
#         classical_params_array,
#         p;
#         lb= learning_problem.p_repr_lb,
#         ub=learning_problem.p_repr_ub)

#     # solve optimization problem
#     sol = solve(prob, BBO_adaptive_de_rand_1_bin(); callback=callback, maxiters=maxiters)
#     final_params_repr = CombiCellModelLearning.reconstruct_learning_params_from_array(sol.minimizer, p_repr_ig, learning_problem.model)
#     final_params_derepr = CombiCellModelLearning.derepresent_all(final_params_repr, learning_problem.model)

#     return final_params_derepr, loss_history
# end

final_params_derepr, loss_history = CombiCellModelLearning.bbo_learn(learning_problem, p_repr_ig)

savedir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/experiments/02102026_bicycleHardAccessory/" # change for diff exp

@save joinpath(savedir, "final_params_derepr.jld2") final_params_derepr
@save joinpath(savedir, "loss_history.jld2") loss_history
@save joinpath(savedir, "model.jld2") model
#TODO : move all the below for hpc later

# # Now use the functions after your optimization:base_path
# println("\n=== Starting Plot Generation ===")

# # Generate fit data once
# p_class = final_params_derepr.p_classical
# # fitData = generate_fit_data(fakeData, p_class, model)

# # Generate all plots and metrics
# all_metrics, fitData = CombiCellModelLearning.generate_all_plots_and_metrics(
#     fakeData, p_class, loss_history, savedir, model
# )
