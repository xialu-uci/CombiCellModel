loaddir = "./cleanData" # modify for hpc
@load joinpath(loaddir, "fakeData.jld2") fakeData
@load joinpath(loaddir, "CombiCell_data.jld2") data

# function convert_params(p_repr, flexi_model)
#   p_class = p_repr.p_classical
#   p_flex = flexi_model.params_repr_ig.flex1_params
#   return ComponentArray(p_classical=p_class, flex1_params=p_flex)
  
# end

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
classdir = mkdir("../CombiCellLocal/experiments/" * exp * "/classical/" * dirName)
flexidir = mkdir("../CombiCellLocal/experiments/" * exp * "/flexi/" * dirName)
model_classical = CombiCellModelLearning.make_ModelCombiClassic(intPoint1= i, intPoint2=j) # defaults 11,12 are the intPoints for fakeData
model_flexi = CombiCellModelLearning.make_ModelCombiFlexi(intPoint1= i, intPoint2=j) # defaults 11,12 are the intPoints for fakeData
# model = CombiCellModelLearning.make_ModelCombiFlexi(intPoint1= i, intPoint2=j) # defaults 11,12 are the intPoints for fakeData

p_repr_ig = deepcopy(model_classical.params_repr_ig)
# learning problem
learning_problem_classical = CombiCellModelLearning.LearningProblem(
    data =data, # fakeData or data (real)
    model= model_classical,
    p_repr_lb=CombiCellModelLearning.represent(model_classical.p_derepresented_lowerbounds, model_classical.intPoints, model_classical),
    p_repr_ub=CombiCellModelLearning.represent(model_classical.p_derepresented_upperbounds, model_classical.intPoints, model_classical),
    mask = trues(realLength), # or fakeLength # no mask for now
    loss_strategy="normalized")



for_simplex_repr, bbo_loss_history = CombiCellModelLearning.bbo_learn(learning_problem_classical, p_repr_ig, model_classical.intPoints)
println("Starting simplex optimization with initial loss: $(bbo_loss_history[end])")
#for_simplex_repr = CombiCellModelLearning.represent(for_simplex_derepr, model.intPoints, model)
for_cmaes_repr, simplex_loss_history = CombiCellModelLearning.simplex_learn(learning_problem_classical, for_simplex_repr, model_classical.intPoints)
   loss_history_classical = vcat(bbo_loss_history, simplex_loss_history)
   final_params_derepr_classical=CombiCellModelLearning.derepresent_all(for_cmaes_repr, model_classical.intPoints, model_classical)
# println("Starting CMA-ES optimization with initial loss: $(simplex_loss_history[end])")
# final_params_repr, cmaes_loss_history = CombiCellModelLearning.cmaes_learn(learning_problem, for_cmaes_repr, model.intPoints; upper_bound_multiplier=10.0)
# loss_history = vcat(bbo_loss_history, simplex_loss_history, cmaes_loss_history)
# final_params_derepr=CombiCellModelLearning.derepresent_all(final_params_repr, model.intPoints, model)

#savedir = "../tempExp" # change for diff exptrues(length(data["x"]))
# savedir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/experiments/02112026_bicycleHardAccessory_realData"
# savedir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/experiments/02112026_bicycleHardAccessory_fakeData"
# savedir ="../CombiCellLocal/experiments/02172026_bicycleHardAccessory_int79_fakeData"
@save joinpath(classdir, "final_params_derepr.jld2") final_params_derepr_classical
@save joinpath(classdir, "loss_history.jld2") loss_history_classical
@save joinpath(classdir, "model.jld2") model_classical
  #  end

# end
