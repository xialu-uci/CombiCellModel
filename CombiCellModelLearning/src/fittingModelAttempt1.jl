
model = CombiCellModelLearning.create_model("ModelCombi_flexO1O2"; flexi_dofs=50)

data = CombiCellModelLearning.get_data()

# mask some rows (train/test split)
shape = size(data["x"]);
mask = trues(shape) # hard code for now
#randomly mask 20% of the data # got to make changes to generate_mask.jl
using Random
Random.seed!(42)
num_elements = prod(shape)
num_mask = round(Int, 0.2 * num_elements)
mask_indices = randperm(num_elements)[1:num_mask]
for idx in mask_indices
    mask[idx] = false
end
train_data = Dict{String, Any}()
test_data = Dict{String, Any}()
for (key, value) in data
    train_data[key] = value[mask]
    test_data[key] = value[.!mask]
end

# print sizes of train and test data
println("Train data size: $(length(train_data["x"]))")
println("Test data size: $(length(test_data["x"]))") 

# ok I think it's good to this point.


# TODO: learn parameters without the existing PulsatileModelLearning code for now
params_repr_ig = deepcopy(model.params_repr_ig)
# TODO: update get_loss(), forward_combi()
params_derepresented = CombiCellModelLearning.derepresent_all(params_repr_ig, model)
O1_pred, O2_pred = CombiCellModelLearning.forward_combi(train_data["x"], params_derepresented)

# vanilla SSR loss on training data
loss_ig = sum((O1_pred .- train_data["O1_00"]) .^ 2) + sum((O2_pred .- train_data["O2_00"]) .^ 2)

println("Initial guess loss: $loss_ig")

