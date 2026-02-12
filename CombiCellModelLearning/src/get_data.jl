
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
