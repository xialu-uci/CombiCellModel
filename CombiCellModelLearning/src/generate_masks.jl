function generate_kfold_masks(shape::Tuple{Int}, k::Int)
    # TODO: modify for combi cell data
    #n_total = prod(shape)  # total number of elements
    n_total = shape[1]
    if n_total % k != 0
        error("Number of elements ($n_total) must be divisible by k ($k)")
    end
    fold_size = div(n_total, k)

    # Create a shuffled vector of all linear indices
    all_indices = shuffle(1:n_total)

    masks = Vector{BitVector}(undef, k) # change BitMatrix to BitVector?
    for i in 1:k
        # Indices to be masked as false (i.e., validation set)
        val_indices = all_indices[(fold_size * (i-1) + 1):(fold_size * i)]

        # Start with all true
        mask = trues(shape)
        # Set validation indices to false
        mask[val_indices] .= false

        masks[i] = mask
    end

    base_path = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/"

    file_path = joinpath(base_path, "assets/kfold_masks.jld2")

    @save file_path masks

    return #masks

    # Example usage:
    #PulsatileModelLearning.generate_kfold_masks((6, 14), 7)

end


function generate_testTrain_masks(shape::Tuple{Int, Int}, test_fraction::Float64)
    n_total = prod(shape)
    n_test = round(Int, n_total * test_fraction)

    # Create a shuffled vector of all linear indices
    all_indices = shuffle(1:n_total)

    test_indices = all_indices[1:n_test]

    # Start with all true
    mask = trues(shape)
    # Set test indices to false
    mask[test_indices] .= false

    base_path = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/"
    file_path = joinpath(base_path, "assets/test_train_mask.jld2")
    @save file_path mask

    return #mask

    # Example usage:
    #PulsatileModelLearning.generate_testTrain_masks((6, 14), 0.2)

end


# Save the masks to a file

function generate_mask(config,data)
    if config["mask_id"] != 0
        # load mask from file
        mask_file_name = "kfold_masks.jld2"
        base_path = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/"
        file_path = joinpath(base_path, "assets/", mask_file_name)
        @load file_path masks
        mask = masks[config["mask_id"]]

        if size(mask) != size(data["x"])
            error("Mask size $(size(mask)) does not match data size $(size(data))")
        end
    else
        # construct mask that excludes some on_time,off_time pairs from SSR
        mask = trues(size(data["x"]))
        for key in data.keys()
            data[key] = data[key][mask] # updated for dict data structure
        end
    end
    return mask
end



# masks = load(joinpath("assets", "kfold_masks.jld2"), "masks")
# @show masks

function create_args(filename::String)

    mask_ids = 1:7
    flexi_dofs = [2,4,8,16,32,64,128,256]


    open(filename, "w") do io
        for a in mask_ids, b in flexi_dofs
            println(io, "--mask_id $a --flexi_dofs $b")
        end
    end
end

# Example usage:
# create_args("batch_scripts/args_cv.txt")