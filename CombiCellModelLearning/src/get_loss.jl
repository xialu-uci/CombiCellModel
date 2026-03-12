
function get_loss(p_repr, intPoints; learning_problem::LearningProblem{M}) where {M<:AbstractModel}
    params_derepresented = derepresent_all(p_repr, intPoints, learning_problem.model)
    x = learning_problem.data["x"]
    KD = learning_problem.data["KD"]
    
    O1_00 = learning_problem.data["O1_00"]
    O2_00 = learning_problem.data["O2_00"]
    O1_10 = learning_problem.data["O1_10"]
    O2_10 = learning_problem.data["O2_10"]
    O1_01 = learning_problem.data["O1_01"]
    O2_01 = learning_problem.data["O2_01"]
    O1_11 = learning_problem.data["O1_11"]
    O2_11 = learning_problem.data["O2_11"]

    # make matrix of all of these
    output_true_matrix = hcat(O1_00, O2_00, O1_10, O2_10, O1_01, O2_01, O1_11, O2_11)
    # o1_true_matrix = hcat(O1_00, O1_10, O1_01, O1_11)
  
    output_pred_matrix = forward_combi(x, KD, params_derepresented, learning_problem.model)

    # Compute loss against learning_problem.O1_data and O2_data
    # ssr = sum((O1_00_pred .- O1_00).^2) + sum((O2_00_pred .- O2_00).^2)
    if learning_problem.loss_strategy == "vanilla"
        ssr = norm(output_true_matrix - output_pred_matrix) # default p =2 (frobenius)
    elseif learning_problem.loss_strategy == "o1_only"
        o1_true_matrix = output_true_matrix[:, [1, 3, 5, 7]]
        o1_pred_matrix = output_pred_matrix[:, [1, 3, 5, 7]]
        ssr = norm(output_true_matrix - output_pred_matrix)

    elseif learning_problem.loss_strategy == "normalized-joint"
        joint_mat = vcat(output_pred_matrix, output_true_matrix)
        joint_max = maximum(abs.(joint_mat), dims = 1)
        output_true_normed = output_true_matrix ./ joint_max
        output_pred_normed = output_pred_matrix ./ joint_max
        ssr = norm(output_true_normed - output_pred_normed);
    elseif learning_problem.loss_strategy == "normalized-true" # normalized with true max only, to avoid issues with very small predicted values dominating the loss
        true_max = maximum(abs.(output_true_matrix), dims = 1)
        output_true_normed = output_true_matrix ./ true_max
        output_pred_normed = output_pred_matrix ./ true_max
        ssr = norm(output_true_normed - output_pred_normed);
    elseif learning_problem.loss_strategy == "normalized-00" # normalized with 00 max only, to avoid issues with very small predicted values dominating the loss
        mat_00 = hcat(O1_00, O2_00)
        max00_mini = maximum(abs.(mat_00), dims = 1)
        # max00 = [max00_mini[1], max00_mini[2], max00_mini[1], max00_mini[2], max00_mini[1], max00_mini[2], max00_mini[1], max00_mini[2]]
        max00 = hcat(max00_mini, max00_mini, max00_mini, max00_mini) # repeat each element 4 times to match the 8 columns of the output matrices
        output_true_normed = output_true_matrix ./ max00
        output_pred_normed = output_pred_matrix ./ max00
        ssr = norm(output_true_normed - output_pred_normed);
    end

    return ssr

end

function get_single_loss(p_repr, intPoints; learning_problem::LearningProblem{M}) where {M<:AbstractModel}
    # TODO: loss strategy must be different from pulsatile model learning
    params_derepresented = derepresent_all(p_repr, intPoints, learning_problem.model)
    x = learning_problem.data["x"]
    KD = learning_problem.data["KD"]
    
    O1 = learning_problem.data["O1"]
    O2 = learning_problem.data["O2"]
    

    # make matrix of all of these
    output_true_matrix = hcat(O1, O2)
  
    output_pred_matrix = forward_combi(x, KD, params_derepresented, learning_problem.model)

    # Compute loss against learning_problem.O1_data and O2_data
    # ssr = sum((O1_00_pred .- O1_00).^2) + sum((O2_00_pred .- O2_00).^2)
    if learning_problem.loss_strategy == "vanilla"
        ssr = norm(output_true_matrix - output_pred_matrix) # default p =2 (frobenius)

    elseif learning_problem.loss_strategy == "o1_only"
        ssr = norm(O1 - output_pred_matrix[:,1])
        
    elseif learning_problem.loss_strategy == "normalized-joint"
        joint_mat = vcat(output_pred_matrix, output_true_matrix)
        joint_max = maximum(abs.(joint_mat), dims = 1)
        output_true_normed = output_true_matrix ./ joint_max
        output_pred_normed = output_pred_matrix ./ joint_max
        ssr = norm(output_true_normed - output_pred_normed);
    elseif learning_problem.loss_strategy == "normalized" # normalized with true max only, to avoid issues with very small predicted values dominating the loss
        true_max = maximum(abs.(output_true_matrix), dims = 1)
        output_true_normed = output_true_matrix ./ true_max
        output_pred_normed = output_pred_matrix ./ true_max
        ssr = norm(output_true_normed - output_pred_normed);
    end

    return ssr

end

# function test_revise()
#     println("test revise")
    
# end


