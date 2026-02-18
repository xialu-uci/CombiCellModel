using Makie
using CairoMakie
using JLD2

loaddir = "./cleanData" # modified for hpc
@load joinpath(loaddir, "fakeData.jld2") fakeData
@load joinpath(loaddir, "CombiCell_data.jld2") data


# savedir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/experiments/02112026_bicycleHardAccessory_realData" # change for diff exp
# savedir = "../CombiCellLocal/experiments/02172026_bicycleHardAccessory_int79_fakeData" # change for diff exp
parentdir = "../CombiCellLocal/experiments/02182026_realData_HPC3"
subdirs = filter(d -> !endswith(d, "logs"), readdir(parentdir, join=true))
for savedir in subdirs

    @load joinpath(savedir, "final_params_derepr.jld2") final_params_derepr
    @load joinpath(savedir, "loss_history.jld2") loss_history
    @load joinpath(savedir, "model.jld2") model

    # println("\n=== Starting Plot Generation ===")


    # Generate fit data once
    p_class = final_params_derepr.p_classical
    # fitData = generate_fit_data(fakeData, p_class, model)

    # Generate all metrics
    fitData = CombiCellModelLearning.generate_fit_data(data, p_class, model)
    all_metrics = CombiCellModelLearning.compute_metrics_per_ligand_condition(data, fitData, savedir)
    # all_metrics, fitData = CombiCellModelLearning.generate_all_plots_and_metrics(
    #     fakeData, p_class, loss_history, savedir, model
    # )   # or change to fake data
end

# summary plots
CombiCellModelLearning.create_metrics_heatmaps(parentdir)