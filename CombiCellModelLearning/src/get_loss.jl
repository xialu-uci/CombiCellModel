
function get_loss(p_repr; learning_problem::LearningProblem{M}) where {M<:AbstractModel}
    # TODO: loss strategy must be different from pulsatile model learning
    # params_derepresented = derepresent_all(p_repr, learning_problem.model)

    # c24_normalized = get_freq_response(
    #     learning_problem.on_times, 
    #     learning_problem.off_times, 
    #     params_derepresented, 
    #     learning_problem.model; 
    #     continuous_pulses=learning_problem.continuous_pulses,    
    #     )

    # if learning_problem.loss_strategy == "vanilla"
    #     # VANILLA SSR
    #     ssr = sum((c24_normalized .- learning_problem.c24_data).^2)
        
    # elseif learning_problem.loss_strategy == "normalized"
    #     # DYNAMIC-RANGE-NORMALIZED SSR
    #     c24_data_ranges_per_on_idx = [
    #         maximum(learning_problem.c24_data[i, :]) - minimum(learning_problem.c24_data[i, :]) for
    #         i in axes(learning_problem.c24_data, 1)
    #     ]
    #     ssr = sum(((c24_normalized .- learning_problem.c24_data) .^ 2) ./ (c24_data_ranges_per_on_idx .^ 2))
    # elseif learning_problem.loss_strategy == "masked"
    #     # MASKED SSR (current default)
    #     c24_data_ranges_per_on_idx = [
    #         maximum(learning_problem.c24_data[i, :]) - minimum(learning_problem.c24_data[i, :]) for
    #         i in axes(learning_problem.c24_data, 1)
    #     ]
    #     c24_data_ranges_per_on_idx_expanded = repeat(c24_data_ranges_per_on_idx, 1, 14)
    #     ssr = sum(((c24_normalized .- learning_problem.c24_data)[learning_problem.mask] .^ 2))
                
    # elseif learning_problem.loss_strategy == "masked_normalized"
    #     # MASKED SSR (current default)
    #     c24_data_ranges_per_on_idx = [
    #         maximum(learning_problem.c24_data[i, :]) - minimum(learning_problem.c24_data[i, :]) for
    #         i in axes(learning_problem.c24_data, 1)
    #     ]
    #     c24_data_ranges_per_on_idx_expanded = repeat(c24_data_ranges_per_on_idx, 1, 14)
    #     ssr = sum(((c24_normalized .- learning_problem.c24_data)[learning_problem.mask] .^ 2) ./ (c24_data_ranges_per_on_idx_expanded[learning_problem.mask] .^ 2))
        
    # elseif learning_problem.loss_strategy == "shifted_masked"
    #     # SYNTHETIC SHIFT TO MEAN FOR FIGURE 2
    #     shifted_data = (
    #         learning_problem.c24_data[learning_problem.mask]
    #          .- mean(learning_problem.c24_data[learning_problem.mask])
    #          .+ mean(c24_normalized[learning_problem.mask])
    #     )
    #     ssr = sum((c24_normalized[learning_problem.mask] .- shifted_data) .^ 2)
        
    # elseif learning_problem.loss_strategy == "masked_rowchi2"
    #     # MASKED ROW CHI-SQUARED with hardcoded standard deviation values per on_time
    #     hardcoded_stdevs_per_on_idx = [
    #         0.04100927692132732,
    #         0.04378366429499135,
    #         0.06384224555400576,
    #         0.05347366120103417,
    #         0.07536453967060187,
    #         0.06202818167874277,
    #     ]
    #     # Square to get variances and take only the rows that exist in the data
    #     variances_for_data = (hardcoded_stdevs_per_on_idx[1:size(learning_problem.c24_data, 1)]) .^ 2
    #     variances_expanded = repeat(variances_for_data, 1, 14)
    #     ssr = sum(((c24_normalized .- learning_problem.c24_data)[learning_problem.mask] .^ 2) ./ (variances_expanded[learning_problem.mask]))
        
    # else
    #     error("Unknown loss_strategy: $(learning_problem.loss_strategy). Options: vanilla, normalized, masked, masked_normalized, shifted_masked, masked_rowchi2")
    # end

    # return ssr

    # attempt 1: new loss strat
    params_derepresented = derepresent_all(p_repr, learning_problem.model)
    O1_pred, O2_pred = forward_combi(learning_problem.x_data, params_derepresented)  # Assume x_data is your input concentrations
    # Compute loss against learning_problem.O1_data and O2_data
    loss = sum((O1_pred .- learning_problem.O1_data).^2) + sum((O2_pred .- learning_problem.O2_data).^2)
    return loss

end

