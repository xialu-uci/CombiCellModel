loaddir = "./cleanData" # modify for hpc
@load joinpath(loaddir, "fakeData.jld2") fakeData
@load joinpath(loaddir, "CombiCell_data.jld2") data

fakeLength = length(fakeData["x"])
realLength = length(data["x"])
# now let's make a classical model and try to fit parameters to the simulated data
# differential evolution
intPoints = ["fI", "alpha", "tT", "g1", "k_on_2d", "kP", "nKP","lamdaX", "nC", "XO1", "O1max", "O2max"]
exp = "03012026_test_realData_flexiO2"
#for i in 1:12
 #   for j in 1:12
# i = parse(Int, ARGS[1])
# j = parse(Int, ARGS[2])

i = 3
j = 3

 # println(i,j)
 # exit
dirName = "cd2" * "-"* intPoints[i] * "-" * "pd1"* "-"* intPoints[j]
savedir = mkdir("../CombiCellLocal/experiments/" * exp * "/" * dirName)
model = CombiCellModelLearning.make_ModelCombiFlexi(intPoint1= i, intPoint2=j) # defaults 11,12 are the intPoints for fakeData

p_repr_ig = deepcopy(model.params_repr_ig)
# learning problem
learning_problem = CombiCellModelLearning.LearningProblem(
    data =data, # fakeData or data (real)
    model= model,
    p_repr_lb=CombiCellModelLearning.represent(model.p_derepresented_lowerbounds, model.intPoints, model),
    p_repr_ub=CombiCellModelLearning.represent(model.p_derepresented_upperbounds, model.intPoints, model),
    mask = trues(realLength), # or fakeLength # no mask for now
    loss_strategy="normalized")



for_simplex_repr, bbo_loss_history = CombiCellModelLearning.bbo_learn(learning_problem, p_repr_ig, model.intPoints)
println("Starting simplex optimization with initial loss: $(bbo_loss_history[end])")
#for_simplex_repr = CombiCellModelLearning.represent(for_simplex_derepr, model.intPoints, model)
for_cmaes_repr, simplex_loss_history = CombiCellModelLearning.simplex_learn(learning_problem, for_simplex_repr, model.intPoints)
#   loss_history = vcat(bbo_loss_history, simplex_loss_history)
#   final_params_derepr=CombiCellModelLearning.derepresent_all(final_params_repr, model.intPoints, model)
println("Starting CMA-ES optimization with initial loss: $(simplex_loss_history[end])")
final_params_repr, cmaes_loss_history = CombiCellModelLearning.cmaes_learn(learning_problem, for_cmaes_repr, model.intPoints; upper_bound_multiplier=10.0)
loss_history = vcat(bbo_loss_history, simplex_loss_history, cmaes_loss_history)
final_params_derepr=CombiCellModelLearning.derepresent_all(final_params_repr, model.intPoints, model)

#savedir = "../tempExp" # change for diff exptrues(length(data["x"]))
# savedir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/experiments/02112026_bicycleHardAccessory_realData"
# savedir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/experiments/02112026_bicycleHardAccessory_fakeData"
# savedir ="../CombiCellLocal/experiments/02172026_bicycleHardAccessory_int79_fakeData"
@save joinpath(savedir, "final_params_derepr.jld2") final_params_derepr
@save joinpath(savedir, "loss_history.jld2") loss_history
@save joinpath(savedir, "model.jld2") model
  #  end

# end
