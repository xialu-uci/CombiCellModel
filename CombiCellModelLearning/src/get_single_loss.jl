
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
    elseif learning_problem.loss_strategy == "normalized"
        joint_mat = vcat(output_pred_matrix, output_true_matrix)
        joint_max = maximum(abs.(joint_mat), dims = 1)
        output_true_normed = output_true_matrix ./ joint_max
        output_pred_normed = output_pred_matrix ./ joint_max
        ssr = norm(output_true_normed - output_pred_normed);
    end

    return ssr

end

# function test_revise()
#     println("test revise")
    
# end


