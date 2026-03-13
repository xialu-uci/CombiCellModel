using CombiCellModelLearning
using ComponentArrays
using Optimization
using OptimizationBBO
using Statistics
using JLD2

loaddir = "./cleanData"
@load joinpath(loaddir, "CombiCell_data_O1only_min0.jld2") data
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

parentdir = "03122026_nonsimultaneous_realData_flexiO1"

expdir = "../CombiCellLocal/experiments/" * parentdir


for cond in conditions
    data_subset = subsets[cond]
    dirName = cond * "_realData"
    classdir = mkdir("../CombiCellLocal/experiments/" * parentdir * "/classical/" * dirName)
    classSimpdir = mkdir("../CombiCellLocal/experiments/" * parentdir * "/classical-simplex/" * dirName)
    flexidir = mkdir("../CombiCellLocal/experiments/" * parentdir * "/flexi/" * dirName)
    model_classical = CombiCellModelLearning.make_ModelCombiClassic() # defaults nothing are the intPoints for fakeData
    model_flexi = CombiCellModelLearning.make_ModelCombiFlexi() # defaults nothing are the intPoints for fakeData
    # model = CombiCellModelLearning.make_ModelCombiFlexi(intPoint1= i, intPoint2=j) # defaults 11,12 are the intPoints for fakeData

    p_repr_ig = deepcopy(model_classical.params_repr_ig)
    # learning problem
    learning_problem_classical = CombiCellModelLearning.LearningProblem(
        data =data_subset, # fakeData or data (real)
        model= model_classical,
        p_repr_lb=CombiCellModelLearning.represent(model_classical.p_derepresented_lowerbounds, model_classical.intPoints, model_classical),
        p_repr_ub=CombiCellModelLearning.represent(model_classical.p_derepresented_upperbounds, model_classical.intPoints, model_classical),
        mask = trues(realLength), # or fakeLength # no mask for now
        loss_strategy="o1_only")



    for_simplex_repr, bbo_loss_history = CombiCellModelLearning.bbo_learn_single(learning_problem_classical, p_repr_ig, model_classical.intPoints)
    final_params_derepr_classical = CombiCellModelLearning.derepresent_all(for_simplex_repr, model_classical.intPoints, model_classical)
    println("Starting simplex optimization with initial loss: $(bbo_loss_history[end])")
    #for_simplex_repr = CombiCellModelLearning.represent(for_simplex_derepr, model.intPoints, model)
    for_cmaes_repr, simplex_loss_history = CombiCellModelLearning.simplex_learn_single(learning_problem_classical, for_simplex_repr, model_classical.intPoints)
    loss_history_classical = bbo_loss_history # save simplex only in flexi loss history for now
    loss_history_classical_simplex = vcat(bbo_loss_history, simplex_loss_history)
    final_params_derepr_classical_simplex=CombiCellModelLearning.derepresent_all(for_cmaes_repr, model_classical.intPoints, model_classical)


    # model_flexi = CombiCellModelLearning.make_ModelCombiFlexi(intPoint1= i, intPoint2=j) # defaults 11,12 are the intPoints for fakeData
    learning_problem_flexi = CombiCellModelLearning.LearningProblem(
        data =data_subset, # fakeData or data (real)
        model= model_flexi,
        p_repr_lb=CombiCellModelLearning.represent(model_flexi.p_derepresented_lowerbounds, model_flexi.intPoints, model_flexi),
        p_repr_ub=CombiCellModelLearning.represent(model_flexi.p_derepresented_upperbounds, model_flexi.intPoints, model_flexi),
        mask = trues(realLength), # or fakeLength # no mask for now
        loss_strategy="o1_only")
    p_repr_flexi = CombiCellModelLearning.convert_params(for_cmaes_repr, model_flexi)
    loss_history_flexi = simplex_loss_history # save simplex only in flexi loss history for now
    for i =1:3 #1:3 works for realData, tried 1:10 for simFlexiData needs even longer
        # global p_repr_flexi, loss_history_flexi
        println("what are intpoints for flexi: $(model_flexi.intPoints)")
        println("Starting CMA-ES optimization with initial loss: $(simplex_loss_history[end])")
        p_repr_flexi, cmaes_loss_i = CombiCellModelLearning.cmaes_learn(learning_problem_flexi, p_repr_flexi, model_flexi.intPoints; upper_bound_multiplier=10.0, single=true)
        push!(loss_history_flexi, cmaes_loss_i...)
        p_repr_flexi, simplex_loss_i = CombiCellModelLearning.simplex_learn_single(learning_problem_flexi, p_repr_flexi, model_flexi.intPoints)
        push!(loss_history_flexi, simplex_loss_i...)
    end
    final_params_derepr_flexi=CombiCellModelLearning.derepresent_all(p_repr_flexi, model_flexi.intPoints, model_flexi)
    # loss_history_flexi = vcat(bbo_loss_history, simplex_loss_history, cmaes_loss_history)
    #savedir = "../tempExp" # change for diff exptrues(length(data["x"]))
    # savedir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/experiments/02112026_bicycleHardAccessory_realData"
    # savedir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/experiments/02112026_bicycleHardAccessory_fakeData"
    # savedir ="../CombiCellLocal/experiments/02172026_bicycleHardAccessory_int79_fakeData"
    @save joinpath(classdir, "final_params_derepr.jld2") final_params_derepr_classical
    @save joinpath(classdir, "loss_history.jld2") loss_history_classical
    @save joinpath(classdir, "model.jld2") model_classical

    model_classical_simplex = deepcopy(model_classical)
    @save joinpath(classSimpdir, "final_params_derepr.jld2") final_params_derepr_classical_simplex
    @save joinpath(classSimpdir, "loss_history.jld2") loss_history_classical_simplex
    @save joinpath(classSimpdir, "model.jld2") model_classical_simplex

    @save joinpath(flexidir, "final_params_derepr.jld2") final_params_derepr_flexi
    @save joinpath(flexidir, "loss_history.jld2") loss_history_flexi
    @save joinpath(flexidir, "model.jld2") model_flexi

  
    all_metrics_class, fitData_class = CombiCellModelLearning.generate_all_plots_single(
        data_subset, final_params_derepr_classical, loss_history_classical, classdir, model_classical
    )


    all_metrics_class_simp, fitData_class_simp = CombiCellModelLearning.generate_all_plots_single(
        data_subset, final_params_derepr_classical_simplex, loss_history_classical_simplex, classSimpdir, model_classical_simplex
        )
    all_metrics_flexi, fitData_flexi = CombiCellModelLearning.generate_all_plots_single(
        data_subset, final_params_derepr_flexi, loss_history_flexi, flexidir, model_flexi
    )

    rmse_normed_dict = Dict{String, Float64}(
        "classical_$cond" => all_metrics_class["RMSE_normed"],
        "classical_simplex_$cond" => all_metrics_class_simp["RMSE_normed"],
        "flexi_$cond" => all_metrics_flexi["RMSE_normed"]
    )

    @save joinpath(expdir, "rmse_normed_dict_$(cond).jld2") rmse_normed_dict

    CombiCellModelLearning.plot_flexi(final_params_derepr_flexi.flex1_params, flexidir)


    # println("\n" * "="^40)
    # println("Condition $cond RMSE Summary")
    # println("="^40)
    # println("  O1 RMSE:       $(round(all_metrics["RMSE_O1"], digits=6))")
    # println("  O2 RMSE:       $(round(all_metrics["RMSE_O2"], digits=6))")
    # println("  Combined RMSE: $(round(all_metrics["RMSE_combined"], digits=6))")
    # println("  Bias:          $(round(all_metrics["bias"], digits=6))")
    # println("="^40 * "\n")
end

# ── RMSE Normalized Summary Table ─────────────────────────────────────────────
all_rmse = Dict{String, Dict{String, Float64}}()
for cond in conditions
    @load joinpath(expdir, "rmse_normed_dict_$(cond).jld2") rmse_normed_dict
    all_rmse[cond] = rmse_normed_dict
end
 
col_width = 22
header_labels = ["Classical", "Classical+Simplex", "Flexi"]
keys_per_cond = ["classical", "classical_simplex", "flexi"]
 
sep = "="^(9 + col_width * 3 + 3)
println("\n" * sep)
println("RMSE Normalized Summary (all conditions)")
println(sep)
println(rpad("Cond", 9) * join([lpad(h, col_width) for h in header_labels]))
println("-"^(9 + col_width * 3 + 3))
for cond in conditions
    row_vals = [get(all_rmse[cond], "$(k)_$(cond)", NaN) for k in keys_per_cond]
    println(rpad(cond, 9) * join([lpad(round(v, digits=6), col_width) for v in row_vals]))
end
println(sep)
 