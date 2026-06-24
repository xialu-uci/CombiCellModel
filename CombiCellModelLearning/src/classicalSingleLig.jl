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
flexi = true # for now
my_reference_rmse = 0.0123

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

# helper functions 
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

# should be transferable to 12x12 simultaneous lig fitting
function learn_save_model(p_repr_ig, savedir, learning_problem, learning_protocol; single = false)
    final_params_repr, loss_history = learning_protocol(learning_problem, p_repr_ig, learning_problem.model.intPoints; single = single)
    final_params_derepr = CombiCellModelLearning.derepresent_all(final_params_repr, learning_problem.model.intPoints, learning_problem.model)
    model = learning_problem.model

    
    @save joinpath(savedir, "final_params_derepr.jld2") final_params_derepr
    @save joinpath(savedir, "loss_history.jld2") loss_history
    @save joinpath(savedir, "model.jld2") model


    return final_params_repr, final_params_derepr, loss_history, model
end

# should be transferable to 12x12 simultaneous lig fitting
function flexi_learn_save(p_repr_flexi, savedir, learning_problem; num_swaps = 3, single = false) 
    loss_history =[]
    for i=1:num_swaps
        # println("check intpoints: $(learning_problem.model.intPoints)")
        # println("Starting CMA-ES optimization with initial loss: $(simplex_loss_history[end])")
        p_repr_flexi, cmaes_loss_i = CombiCellModelLearning.cmaes_learn(learning_problem, p_repr_flexi, learning_problem.model.intPoints; upper_bound_multiplier=10.0, single=single)
        push!(loss_history, cmaes_loss_i...)
        p_repr_flexi, simplex_loss_i = CombiCellModelLearning.simplex_learn(learning_problem, p_repr_flexi, learning_problem.model.intPoints; single = single)
        push!(loss_history, simplex_loss_i...)
        println("Flexi fitting loss after cycle $i: $(loss_history[end])")
    end
    final_params_derepr = CombiCellModelLearning.derepresent_all(p_repr_flexi, learning_problem.model.intPoints, learning_problem.model)
    model = learning_problem.model

    @save joinpath(savedir, "final_params_derepr.jld2") final_params_derepr
    @save joinpath(savedir, "loss_history.jld2") loss_history
    @save joinpath(savedir, "model.jld2") model

    return final_params_derepr, loss_history, model
end

# helper function: save to .txt a table of RMSEs (not normed) for classical-DE, classical-DE-NM, and flexi for each ligand condition
function rmse_table_1lig(parentdir, conditions, ref)
    classical_dir = joinpath(parentdir, "classical-DE")
    simplex_dir   = joinpath(parentdir, "classical-DE-NM")
    flexi_dir     = joinpath(parentdir, "flexi-DE-NM-CMAES")
    include_flexi = isdir(flexi_dir)

    # helper: load RMSE for a given condition subdirectory
    function load_rmse(dir, cond)
        metrics_path = joinpath(dir, cond, "model_metrics_single.jld2")
        if isfile(metrics_path)
            @load metrics_path metrics_dict
            return metrics_dict["RMSE"]
        else
            return NaN
        end
    end

    col_headers = include_flexi ?
        ["Condition", "Classical-DE", "Classical-DE-NM", "Flexi-DE-NM-CMAES"] :
        ["Condition", "Classical-DE", "Classical-DE-NM"]
    col_width = 20

    lines = String[]
    push!(lines, "RMSE Summary (reference RMSE = $(round(ref, digits=6)))")
    push!(lines, "="^(col_width * length(col_headers)))
    push!(lines, join(rpad.(col_headers, col_width)))
    push!(lines, "-"^(col_width * length(col_headers)))

    for cond in conditions
        rmse_classical = load_rmse(classical_dir, cond)
        rmse_simplex   = load_rmse(simplex_dir,   cond)
        vals = [cond, string(round(rmse_classical, digits=6)), string(round(rmse_simplex, digits=6))]
        if include_flexi
            rmse_flexi = load_rmse(flexi_dir, cond)
            push!(vals, string(round(rmse_flexi, digits=6)))
        end
        push!(lines, join(rpad.(vals, col_width)))
    end

    push!(lines, "="^(col_width * length(col_headers)))

    out_path = joinpath(parentdir, "rmse_summary.txt")
    open(out_path, "w") do f
        foreach(line -> println(f, line), lines)
    end
    println("RMSE table saved to $out_path")
end

# compute reference rmses

# classical training
for cond in conditions
    data_subset = subsets[cond]
   
    savedir_classical, model_classical, learning_problem_classical = set_up_1lig_model(classical_dir, cond, CombiCellModelLearning.make_ModelCombiClassic, data_subset)

    p_repr_ig = deepcopy(model_classical.params_repr_ig)

    final_params_repr_classical, final_params_derepr_classical, loss_history_classical, model_classical = learn_save_model(p_repr_ig, savedir_classical, learning_problem_classical, CombiCellModelLearning.bbo_learn, single = true)
    # plotting, getting metrics for classical
  
    all_metrics_classical, fitData_classical = CombiCellModelLearning.generate_all_plots_single(
        data_subset, final_params_derepr_classical, loss_history_classical, savedir_classical, model_classical; o1_only = true
    )

    println("\n" * "="^40)
    println("Classical (DE) Condition $cond RMSE Summary")
    println("="^40)
    println("  RMSE      $(round(all_metrics_classical["RMSE"], digits=6))")
    println("  Normed RMSE:       $(round(all_metrics_classical["RMSE_normed"], digits=6))")
    #  println("  Combined RMSE: $(round(all_metrics["RMSE_combined"], digits=6))")
    println("  Bias:          $(round(all_metrics_classical["bias"], digits=6))")

    if outputs[1] && outputs[2]
        println("  O1 RMSE      $(round(all_metrics_classical["RMSE_O1"], digits=6))")
        println("  O2 RMSE      $(round(all_metrics_classical["RMSE_O2"], digits=6))")
    end
    println("="^40 * "\n")

    # simplex training
    println("Starting simplex optimization with initial loss: $(loss_history_classical[end])")

    savedir_simplex, model_simplex, learning_problem_simplex = set_up_1lig_model(simplex_dir, cond, CombiCellModelLearning.make_ModelCombiClassic, data_subset)
    final_params_repr_simplex, final_params_derepr_simplex, loss_history_simplex, model_simplex = learn_save_model(final_params_repr_classical, savedir_simplex, learning_problem_simplex, CombiCellModelLearning.simplex_learn, single = true)
    all_metrics_simplex, fitData_simplex = CombiCellModelLearning.generate_all_plots_single(
        data_subset, final_params_derepr_simplex, loss_history_simplex, savedir_simplex, model_simplex; o1_only = true
    )
    # if flexi, then flexi training
    println("what are intpoints for classical: $(model_classical.intPoints)")
    

    savedir_flexi, model_flexi, learning_problem_flexi = set_up_1lig_model(flexi_dir, cond, CombiCellModelLearning.make_ModelCombiFlexi_O1, data_subset) #Flexi_O1 means that the eqn that is flexi is the O1 eqn
    println("what are intpoints for flexi: $(model_flexi.intPoints)")
    println("Starting CMA-ES optimization with initial loss: $(loss_history_simplex[end])")
        
    p_repr_flexi = CombiCellModelLearning.convert_params(final_params_repr_simplex, model_flexi)

    final_params_derepr_flexi, loss_history_flexi, model_flexi = flexi_learn_save(p_repr_flexi, savedir_flexi, learning_problem_flexi; single = true)
    all_metrics_flexi, fitData_flexi = CombiCellModelLearning.generate_all_plots_single(
        data_subset, final_params_derepr_flexi, loss_history_flexi, savedir_flexi, model_flexi; o1_only = true
    )

    CombiCellModelLearning.plot_flexi(final_params_derepr_flexi.flex1_params, savedir_flexi)


end

rmse_table_1lig(parentdir, conditions, my_reference_rmse)


