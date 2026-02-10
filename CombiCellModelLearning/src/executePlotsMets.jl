using Makie
using CairoMakie
using JLD2

loaddir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/data/" # modify for hpc
@load joinpath(loaddir, "fakeData.jld2") fakeData
# @load joinpath(loaddir, "CombiCell_data.jld2") data


savedir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/experiments/02102026_bicycleHardAccessory_realData/" # change for diff exp

@load joinpath(savedir, "final_params_derepr.jld2") final_params_derepr
@load joinpath(savedir, "loss_history.jld2") loss_history
@load joinpath(savedir, "model.jld2") model

println("\n=== Starting Plot Generation ===")


# Generate fit data once
p_class = final_params_derepr.p_classical
# fitData = generate_fit_data(fakeData, p_class, model)

# Generate all plots and metrics
all_metrics, fitData = CombiCellModelLearning.generate_all_plots_and_metrics(
    fakeData, p_class, loss_history, savedir, model
)