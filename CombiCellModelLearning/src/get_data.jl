
# Run it in the repl:
# CombiCellModelLearning.get_data(save_data=true)

function get_data(; save_data=false)
   

    base_path = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/"

    println(base_path)

    println((joinpath(base_path, "data/cleanedCD69_IL8.csv")))

    # For CombiCell model no accessory: inputs are x and KD, outputs are O1_00 and O2_00

    # Read the CSV file
    df = DataFrame(CSV.File(joinpath(base_path, "data/cleanedCD69_IL8.csv")))

    # Oa_bc <- a = 1 or 2 (which output type), b = 0 or 1 (w or w/out CD58), c = 0 or 1 (w or w/out PDL1)
    # if zeroed =
    # Extract the relevant columns
    data = Dict(
        "x" => df.x,
        "KD" => df.KD,
        "O1_00" => df.O1_00* .01,
        "O1_10" => df.O1_10* .01,
        "O1_01" => df.O1_01* .01,
        "O1_11" => df.O1_11* .01,
        "O2_00" => df.O2_00,
        "O2_10" => df.O2_10,
        "O2_01" => df.O2_01,
        "O2_11" => df.O2_11,
    )

    if save_data
        @save joinpath(base_path, "data/CombiCell_data.jld2") data
    end

    return data

end

function get_data_min0(; save_data=false)
    data = get_data(save_data=false) # load the data without saving, since we will save the min0 version at the end of this function
    # for each kd, for each ligand condition, subtract minimum output value across all x values for that kd and ligand condition from the output values for that kd and ligand condition
    ligand_conds = ["00", "10", "01", "11"]
    for ligand_cond in ligand_conds
        for output_type in ["O1", "O2"]
            col_name = output_type * "_" * ligand_cond
            for kd in unique(data["KD"])
                idx = findall(data["KD"] .== kd)
                min_val = minimum(data[col_name][idx])
                data[col_name][idx] .-= min_val
            end
        end
    end
 

    if save_data
        base_path = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/"
        @save joinpath(base_path, "data/CombiCell_data_min0.jld2") data
    end

    return data
end

function get_data_O1(; save_data = false)
    data = get_data_min0(save_data = false)
    ligand_conds = ["00", "10", "01", "11"]
    for ligand_cond in ligand_conds
        col_name_O2 = "O2" * "_" * ligand_cond
        data[col_name_O2] = zeros(length(data[col_name_O2]))
    end
 

    if save_data
        base_path = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/"
        @save joinpath(base_path, "data/CombiCell_data_O1only_min0.jld2") data
    end

    return data
    


end
