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
#   (2) parentdir #DONE: would be useful to makedirs if they don't already exist. print warnings if they already exist though (avoid overwriting old files on accident)
#   (3) DONE: add some reference for which outputs are included in the data
#       (a) DONE: implement using this tag (add keyword argument to makeModel__())
#           (iii) DONE: try everything first with just modifications to classical model and training.
#   (4) DONE: add some reference for if I want to flexi fit
#       (b) TODO: implement using this tag (wrap stuff in if blocks)



parentdir = "../CombiCellLocal/experiments/06182026_sepLigFits" 
data_file = "CombiCell_data_O1only_min0_noKD162.jld2"
outputs = [true, false] # or [true, false] (for now, want flexibility to add [false, true] later)
flexi = false # for now

# make parentdir if it doesn't exist yet, give warning if it already exists
if isdir(parentdir)
    @warn "Directory '$parentdir' already exists. Reusing it."
else
    mkdir(parentdir)
end

# make parentdir/classical and parentdir/classical-simplex
classical_dir = joinpath(parentdir, "classical-DE")
simplex_dir = joinpath(parentdir, "classical-DE-NM")

isdir(classical_dir) || mkdir(classical_dir)
isdir(simplex_dir) || mkdir(simplex_dir)

# if flexi, make parentdir/flexi
if flexi
    flexi_dir = joinpath(parentdir, "flexi-DE-NM-CMAES")
    isdir(flexi_dir) || mkdir(flexi_dir)
end


# load data
loaddir = "./cleanData"
@load joinpath(loaddir, data_file) data
realLength = length(data["x"])

# define loss strategy

my_loss_strategy = "normalized"
if !outputs[2]
    my_loss_strategy = "o1_only"
end

# ligand conditions:
#   00 = no accessory
#   10 = cd2 only (upregs tcr activation)
#   01 = pd1 only (downregs tcr activation)
#   11 = both
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

# make a function for setting up savedir, making the model, making the learning problem
function set_up_1lig_model(dir, cond, maker, data_subset)
    # data_subset = subsets[cond]
    savedir = joinpath(dir, cond)
    isdir(savedir) || mkdir(savedir)
    model = maker(; output1 = outputs[1], output2 = outputs[2])
    learning_problem = CombiCellModelLearning.LearningProblem(data = data_subset,
        model = model,
        p_repr_lb=CombiCellModelLearning.represent(model.p_derepresented_lowerbounds, model.intPoints, model),
        p_repr_ub=CombiCellModelLearning.represent(model.p_derepresented_upperbounds, model.intPoints, model),
        mask=trues(realLength),
        loss_strategy=my_loss_strategy
    )
    return savedir, model, learning_problem
end

# classical training
for cond in conditions
    data_subset = subsets[cond]
    # # dirName = cond * data_description
    # savedir = joinpath(classical_dir, cond)
    # isdir(savedir) || mkdir(savedir)
    # model_classical = CombiCellModelLearning.make_ModelCombiClassic(; output1 = outputs[1], output2 = outputs[2])
    # p_repr_ig = deepcopy(model_classical.params_repr_ig)

    # learning_problem_classical = CombiCellModelLearning.LearningProblem(
    #     data=data_subset,
    #     model=model_classical,
    #     p_repr_lb=CombiCellModelLearning.represent(model.p_derepresented_lowerbounds, model.intPoints, model),
    #     p_repr_ub=CombiCellModelLearning.represent(model.p_derepresented_upperbounds, model.intPoints, model),
    #     mask=trues(realLength),
    #     loss_strategy=my_loss_strategy)
    savedir_classical, model_classical, learning_problem_classical = set_up_1lig_model(classical_dir, cond, CombiCellModelLearning.make_ModelCombiClassic, data_subset)

    p_repr_ig = deepcopy(model_classical.params_repr_ig)

    classical_params_repr, classical_loss_history = CombiCellModelLearning.bbo_learn_single(learning_problem_classical, p_repr_ig, model_classical.intPoints)
    classical_params_derepr = CombiCellModelLearning.derepresent_all(classical_params_repr, model_classical.intPoints, model_classical)

    @save joinpath(savedir_classical, "final_params_derepr.jld2") classical_params_derepr
    @save joinpath(savedir_classical, "loss_history.jld2") classical_loss_history
    @save joinpath(savedir_classical, "model.jld2") model_classical

    # plotting, getting metrics for classical
    p_class = classical_params_derepr.p_classical
    all_metrics, fitData = CombiCellModelLearning.generate_all_plots_single(
        data_subset, p_class, classical_loss_history, savedir_classical, model_classical; o1_only = true
    )

    println("\n" * "="^40)
    println("Classical (DE) Condition $cond RMSE Summary")
    println("="^40)
    println("  RMSE      $(round(all_metrics["RMSE"], digits=6))")
    println("  Normed RMSE:       $(round(all_metrics["RMSE_normed"], digits=6))")
    #  println("  Combined RMSE: $(round(all_metrics["RMSE_combined"], digits=6))")
    println("  Bias:          $(round(all_metrics["bias"], digits=6))")

    if outputs[1] && outputs[2]
        println("  O1 RMSE      $(round(all_metrics["RMSE_O1"], digits=6))")
        println("  O2 RMSE      $(round(all_metrics["RMSE_O2"], digits=6))")
    end
    println("="^40 * "\n")

# simplex training

# if flexi, then flexi training
end

