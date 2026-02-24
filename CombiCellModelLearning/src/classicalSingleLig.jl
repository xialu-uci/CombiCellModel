using CombiCellModelLearning
using ComponentArrays # i feel like I shouldn't need this in here...
using Optimization
using OptimizationBBO
using Statistics
using JLD2


loaddir = "./cleanData" # modify for hpc
@load joinpath(loaddir, "CombiCell_data.jld2") data
realLength = length(data["x"])


# subset data
conditions = ["00", "10", "01", "11"]
subsets = Dict{String, Dict{String, Vector{Float64}}}() # let's make a dict of dicts
    
for cond in conditions
    subsets[cond] = Dict(
        "x"           => data["x"],
        "KD"          => data["KD"],
        "O1"  => data["O1_$(cond)"],
        "O2"  => data["O2_$(cond)"]
    )
end
    
data_00 = subsets["00"]
data_10 = subsets["10"]
data_01 = subsets["01"]
data_11 = subsets["11"]

# start dir
parentdir = "02232026_nonsimultaneous_realData"
# let's train for each
for cond in conditions
    data_subset = subsets[cond]
    dirName = cond *"_realData"
    savedir = mkdir("../CombiCellLocal/experiments/" * parentdir * "/" * dirName)
    model = CombiCellModelLearning.make_ModelCombiClassic() # defaults nothing

    p_repr_ig = deepcopy(model.params_repr_ig)
    # learning problem
    learning_problem = CombiCellModelLearning.LearningProblem(
        data =data_subset, # fakeData or data (real)
        model= model,
        p_repr_lb=CombiCellModelLearning.represent(model.p_derepresented_lowerbounds, model.intPoints, model),
        p_repr_ub=CombiCellModelLearning.represent(model.p_derepresented_upperbounds, model.intPoints, model),
        mask = trues(realLength), # or fakeLength # no mask for now
        loss_strategy="normalized")



        final_params_derepr, loss_history = CombiCellModelLearning.bbo_learn_single(learning_problem, p_repr_ig, model.intPoints)


        @save joinpath(savedir, "final_params_derepr.jld2") final_params_derepr
        @save joinpath(savedir, "loss_history.jld2") loss_history
        @save joinpath(savedir, "model.jld2") model
end



