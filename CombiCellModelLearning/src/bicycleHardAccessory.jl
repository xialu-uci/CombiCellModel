# Run from CombiCellModel
using CombiCellModelLearning
using ComponentArrays # i feel like I shouldn't need this in here...
using Optimization
using OptimizationBBO
using Statistics
using JLD2
# using Makie
# using CairoMakie



loaddir = "./cleanData" # modify for hpc
@load joinpath(loaddir, "fakeData.jld2") fakeData
@load joinpath(loaddir, "CombiCell_data.jld2") data

fakeLength = length(fakeData["x"])
realLength = length(data["x"])
# now let's make a classical model and try to fit parameters to the simulated data
# differential evolution


model = CombiCellModelLearning.make_ModelCombiClassic()

p_repr_ig = deepcopy(model.params_repr_ig)
# learning problem
learning_problem = CombiCellModelLearning.LearningProblem(
     data =fakeData, # fakeData or data (real)
     model= model,
     p_repr_lb=CombiCellModelLearning.represent(model.p_derepresented_lowerbounds, model),
     p_repr_ub=CombiCellModelLearning.represent(model.p_derepresented_upperbounds, model),
     mask = trues(fakeLength), # or fakeLength # no mask for now
     loss_strategy="normalized")



final_params_derepr, loss_history = CombiCellModelLearning.bbo_learn(learning_problem, p_repr_ig)

#savedir = "../tempExp" # change for diff exptrues(length(data["x"]))
# savedir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/experiments/02112026_bicycleHardAccessory_realData"
# savedir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/experiments/02112026_bicycleHardAccessory_fakeData"
savedir ="../CombiCellLocal/experiments/02152026_bicycleHardAccessory_fakeData"
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
