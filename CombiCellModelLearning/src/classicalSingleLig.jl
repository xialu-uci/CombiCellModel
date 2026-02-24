using CombiCellModelLearning
using ComponentArrays
using Optimization
using OptimizationBBO
using Statistics
using JLD2

loaddir = "./cleanData"
@load joinpath(loaddir, "CombiCell_data.jld2") data
realLength = length(data["x"])

conditions = ["00", "10", "01", "11"]
subsets = Dict{String, Dict{String, Vector{Float64}}}()
for cond in conditions
    subsets[cond] = Dict(
        "x"  => data["x"],
        "KD" => data["KD"],
        "O1" => data["O1_$(cond)"],
        "O2" => data["O2_$(cond)"]
    )
end

parentdir = "02242026_nonsimultaneous_realData"

for cond in conditions
    data_subset = subsets[cond]
    dirName = cond * "_realData"
    savedir = mkdir("../CombiCellLocal/experiments/" * parentdir * "/" * dirName)
    model = CombiCellModelLearning.make_ModelCombiClassic()
    p_repr_ig = deepcopy(model.params_repr_ig)

    learning_problem = CombiCellModelLearning.LearningProblem(
        data=data_subset,
        model=model,
        p_repr_lb=CombiCellModelLearning.represent(model.p_derepresented_lowerbounds, model.intPoints, model),
        p_repr_ub=CombiCellModelLearning.represent(model.p_derepresented_upperbounds, model.intPoints, model),
        mask=trues(realLength),
        loss_strategy="normalized")

    final_params_derepr, loss_history = CombiCellModelLearning.bbo_learn_single(learning_problem, p_repr_ig, model.intPoints)

    @save joinpath(savedir, "final_params_derepr.jld2") final_params_derepr
    @save joinpath(savedir, "loss_history.jld2") loss_history
    @save joinpath(savedir, "model.jld2") model

    p_class = final_params_derepr.p_classical
    all_metrics, fitData = CombiCellModelLearning.generate_all_plots_single(
        data_subset, p_class, loss_history, savedir, model
    )

    println("\n" * "="^40)
    println("Condition $cond RMSE Summary")
    println("="^40)
    println("  O1 RMSE:       $(round(all_metrics["RMSE_O1"], digits=6))")
    println("  O2 RMSE:       $(round(all_metrics["RMSE_O2"], digits=6))")
    println("  Combined RMSE: $(round(all_metrics["RMSE_combined"], digits=6))")
    println("  Bias:          $(round(all_metrics["bias"], digits=6))")
    println("="^40 * "\n")
end