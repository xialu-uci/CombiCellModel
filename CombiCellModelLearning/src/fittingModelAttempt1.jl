using JLD2
using CombiCellModelLearning
model = CombiCellModelLearning.create_model("ModelCombi_flexO1O2"; flexi_dofs=50)

data = CombiCellModelLearning.get_data()

# mask some rows (train/test split)
# shape = size(data["x"]);
# mask = trues(shape) # hard code for now
# #randomly mask 20% of the data # got to make changes to generate_mask.jl # is this ok instead of k fold?
# using Random
# Random.seed!(42)
# num_elements = prod(shape)
# num_mask = round(Int, 0.2 * num_elements)
# mask_indices = randperm(num_elements)[1:num_mask]
# for idx in mask_indices
#     mask[idx] = false
# end
# train_data = Dict{String, Any}()
# test_data = Dict{String, Any}()
# for (key, value) in data
#     train_data[key] = value[mask]
#     test_data[key] = value[.!mask]
# end

# # print sizes of train and test data
# println("Train data size: $(length(train_data["x"]))")
# println("Test data size: $(length(test_data["x"]))") 

# ok I think it's good to this point.


# TODO: learn parameters without the existing PulsatileModelLearning code for now
params_repr_ig = deepcopy(model.params_repr_ig)
# TODO: update get_loss(), forward_combi() (forward combi maybe ok)
params_derepresented = CombiCellModelLearning.derepresent_all(params_repr_ig, model)
O1_pred, O2_pred = CombiCellModelLearning.forward_combi(train_data["x"], train_data["KD"], params_derepresented)

# vanilla SSR loss on training data
loss_ig = sum((O1_pred .- train_data["O1_00"]) .^ 2) + sum((O2_pred .- train_data["O2_00"]) .^ 2)



println("Initial guess loss: $loss_ig")

# will need for loop with some rounds of cmaes and some rounds of simplex
# using CMAEvolutionStrategy

# TODO: will want to use .load_cmaes_hyperparams from PulsatileModelLearning.jl? having difficulty finding
# CMAES hyperparameters
    # "cmaes_hyperparams" => Dict(
        # "c_1" => 0.0,
        # "c_c" => 0.0,
        # "c_mu" => 0.0,
        # "c_sigma" => 0.0,
        # "c_m" => 1.0,
       # "sigma0" => -1,  # Use -1 for adaptive sigma0 computation
        # "weights" => nothing  # Optional
# defaults: CMAESProtocol(; maxiters=10, mu=40, lambda=100, upper_bound_multiplier=10.0,
              # c_1=0.0, c_c=0.0, c_mu=0.0, c_sigma=0.0, c_m=1.0, sigma0=-1, weights=nothing) =
   # CMAESProtocol(maxiters, mu, lambda, upper_bound_multiplier, c_1, c_c, c_mu, c_sigma, c_m, sigma0, weights)

#  base_bath = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/data/"


    # Example usage:
    #PulsatileModelLearning.generate_kfold_masks((6, 14), 7)


# now let's see if we can generate k fold masks and load them properly
CombiCellModelLearning.generate_kfold_masks(shape, 6) # 72 div by 6 
# ok yes that worked

# now make a config?


# mask = CombiCellModelLearning.generate_mask

cmaes_protocol = CombiCellModelLearning.CMAESProtocol()

# function learn(protocol::CMAESProtocol, learning_problem, p_repr_ig; 
               # logging::Bool=false, callback_config::CallbackConfig=CallbackConfig())
    

result = CombiCellModelLearning.learn(cmaes_protocol, train_data, params_repr_ig, model)
best_params, loss_history = result["parameters"], result["loss_history"]
println("Final loss after CMA-ES: $(loss_history[end])")
# plot loss history
using Plots
plot(loss_history, yscale=:log10, xlabel="Iteration", ylabel="Loss", title="CMA-ES Loss History");

