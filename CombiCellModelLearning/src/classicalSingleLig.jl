using CombiCellModelLearning
using ComponentArrays
using Optimization
using OptimizationBBO
using Statistics
using JLD2

#(A) TODO: which files can be combined into 1 file?
#   (1) I think I can combine all the SingleLig files into one (classicalSingleLig.jl, flexiSingleLig.jl, flexiSingleLigO1.jl)
#   (2) need to add some "configuration" info at the beginning explicating which kinds of training to run --> see "todo" tags in (B3-4)
#       (a) every time I do flexi, I do also do classical first and end up saving both --> need a tag to say if I want to "continue" to flexi training
#       (b) both classical and flexi will be modified based on which outputs are being used for fitting
#(B) TODO: which things do I modify each time? 
#   (1) data loading
#   (2) parentdir #TODO: would be useful to makedirs if they don't already exist. print warnings if they already exist though (avoid overwriting old files on accident)
#   (3) DONE: add some reference for which outputs are included in the data
#       (a) TODO: implement using this tag (add keyword argument to makeModel__())
#           (iii) TODO: try everything first with just modifications to classical model and training.
#   (4) DONE: add some reference for if I want to flexi fit
#       (b) TODO: implement using this tag (wrap stuff in if blocks)



parentdir = "02242026_nonsimultaneous_realData" 
data_file = "CombiCell_data.jld2"
outputs = [true, true] # or [true, false] (for now, want flexibility to add [false, true] later)
flexi = false # for now
loaddir = "./cleanData"
@load joinpath(loaddir, data_file) data
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

    final_params_repr, loss_history = CombiCellModelLearning.bbo_learn_single(learning_problem, p_repr_ig, model.intPoints)
    final_params_derepr = CombiCellModelLearning.derepresent_all(final_params_repr, model.intPoints, model)

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