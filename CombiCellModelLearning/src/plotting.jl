

function generate_fit_data(fakeData, p_class, model)
    """
    Generates model predictions for all outputs using fitted parameters.
    """
    # Extract input data
    x_data = fakeData["x"]
    kD_data = fakeData["KD"]
    
    predictions = CombiCellModelLearning.forward_combi(x_data, kD_data, p_class, model)
    
    O1_00_fit = predictions[:, 1]
    O2_00_fit = predictions[:, 2]
    O1_10_fit = predictions[:, 3]
    O2_10_fit = predictions[:, 4]
    O1_01_fit = predictions[:, 5]
    O2_01_fit = predictions[:, 6]
    O1_11_fit = predictions[:, 7]
    O2_11_fit = predictions[:, 8]
    
    # Create fitData dictionary
    fitData = Dict{String, Vector{Float64}}(
        "x" => x_data,
        "KD" => kD_data,
        "O1_00" => O1_00_fit,
        "O2_00" => O2_00_fit,
        "O1_10" => O1_10_fit,
        "O2_10" => O2_10_fit,
        "O1_01" => O1_01_fit,
        "O2_01" => O2_01_fit,
        "O1_11" => O1_11_fit,
        "O2_11" => O2_11_fit
    )
    
    return fitData
end

#  loss w makie
function plot_loss_history(loss_history, savedir)
    """
    Plots the loss history from the optimization process using Makie.
    """
    fig = Figure(size=(800, 600))
    ax = Makie.Axis(fig[1, 1],
              xlabel="Iteration",
              ylabel="Loss",
              title="Loss History During Optimization",
              yscale=log10)
    
    lines!(ax, 1:length(loss_history), loss_history, 
           linewidth=2, color=:blue, label="Loss")
    
    # Add legend
    axislegend(ax, position=:rt)
    
    # Add grid
    ax.xgridvisible = true
    ax.ygridvisible = true
    ax.xgridstyle = :dash
    ax.ygridstyle = :dash
    
    # Save figure
    save_path = joinpath(savedir, "loss_history_makie.png")
    save(save_path, fig)
    println("Loss history plot saved to: $save_path")
    
    return fig
end

# 2. Plot fit vs data using Makie
# 2. Plot fit vs data using Makie - with separate figures for each kD
function plot_fit_vs_data(dataTrue, fitData, savedir)
    """
    Creates 3 sets of plots comparing fitted model predictions to simulated data,
    one set for each unique kD value.
    """
    x_data = dataTrue["x"]
    kD_data = dataTrue["KD"]
    output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    
    # Get unique kD values and their indices
    unique_kDs = unique(kD_data)
    println("Found $(length(unique_kDs)) unique kD values: $unique_kDs")

    # Filter out zero values from x_data for log scale
    valid_global_idx = x_data .> 0
    x_nonzero = x_data[valid_global_idx]
    println("\nGlobal x range (non-zero): $(minimum(x_nonzero)) to $(maximum(x_nonzero))")
    println("log10(x) range: $(log10(minimum(x_nonzero))) to $(log10(maximum(x_nonzero)))")
    
    
    # Create a figure for each unique kD
    figures = []
    
    for (kD_idx, kD_value) in enumerate(unique_kDs)
        # Find indices for this kD value
        kD_indices = findall(kD_data .== kD_value)

        # Filter indices to only include non-zero x values
        valid_kD_idx = [i for i in kD_indices if x_data[i] > 0]
        
        
        # Create figure with 4x2 grid for this kD
        fig = Figure(size=(1400, 1800))
        
        for (output_idx, output_name) in enumerate(output_names)
            # Calculate row and column
            row = ((output_idx-1) ÷ 2) + 1
            col = ((output_idx-1) % 2) + 1
            
            # Extract data and fit for this kD only
            data = dataTrue[output_name][valid_kD_idx]
            fit = fitData[output_name][valid_kD_idx]
            x_subset = x_data[valid_kD_idx]
            
            # Sort by x for smooth line plot (fixes polygons)
            sort_idx = sortperm(x_subset)
            x_sorted = x_subset[sort_idx]
            fit_sorted = fit[sort_idx]
            
            # Create axis for this subplot
            ax = Makie.Axis(fig[row, col],
                      xlabel=(row == 4 ? "x" : ""),
                      ylabel=output_name,
                      title="kD = $kD_value: $output_name",
                      xscale=log10
                      )  # Using log scale for x 
            
            # Plot data points
            scatter!(ax, x_subset, data,
                     markersize=8,
                     color=(:steelblue, 0.7),
                     label="Data",
                     marker=:diamond)
            
            # Plot fitted curve
            lines!(ax, x_sorted, fit_sorted,
                   linewidth=2.5,
                   color=:red,
                   label="Fit")
            
            # Add legend to first plot only to save space
            if output_idx == 1
                axislegend(ax, position=:lt)
            end
            
            # Add grid
            ax.xgridvisible = true
            ax.ygridvisible = true
            ax.xminorgridvisible = true
        end
        
        # Add overall title for this figure
        Label(fig[0, :], "Model Fit vs Data - kD = $kD_value", 
              fontsize=20, font=:bold)
        
        # Adjust layout
        colgap!(fig.layout, 20)
        rowgap!(fig.layout, 20)
        
        # Save this figure
        save_path = joinpath(savedir, "fit_vs_data_kD_$(kD_value).png")
        save(save_path, fig)
        println("Saved fit vs data plot for kD = $kD_value to: $save_path")
        
        push!(figures, fig)
    end
    
    
    return figures
end



# # 3. Plot error using Makie
# function plot_error_all_outputs(fakeData, fitData, savedir)
#     """
#     Creates 8 error plots showing absolute differences between data and fit using Makie.
#     """
#     x_data = fakeData["x"]
#     output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    
#     # Create figure with 4x2 grid
#     fig = Figure(resolution=(1400, 1800))
    
#     for (idx, output_name) in enumerate(output_names)
#         row = ((idx-1) ÷ 2) + 1
#         col = ((idx-1) % 2) + 1
        
#         # Extract data and fit
#         data = fakeData[output_name]
#         fit = fitData[output_name]
        
#         # Calculate absolute error
#         error = abs.(data .- fit)
#         mean_error = mean(error)
        
#         # Create axis
#         ax = Makie.Axis(fig[row, col],
#                   xlabel=(row == 4 ? "x" : ""),
#                   ylabel="|Data - Fit|",
#                   title="Error: $output_name")
        
#         # Plot error points
#         scatter!(ax, x_data, error,
#                  markersize=6,
#                  color=:red,
#                  label="Absolute Error",
#                  marker=:diamond)
        
#         # Add mean error line
#         hlines!(ax, [mean_error],
#                 color=:blue,
#                 linewidth=2,
#                 linestyle=:dash,
#                 label="Mean Error = $(round(mean_error, digits=5))")
        
#         # Add legend to first plot only
#         if idx == 1
#             axislegend(ax, position=:rt)
#         end
        
#         # Add grid
#         ax.xgridvisible = true
#         ax.ygridvisible = true
#     end
    
#     # Add overall title
#     Label(fig[0, :], "Error Plots (All Outputs)", 
#           fontsize=20, font=:bold)
    
#     # Adjust layout
#     colgap!(fig.layout, 20)
#     rowgap!(fig.layout, 20)
    
#     save_path = joinpath(savedir, "error_plots_all_outputs_makie.png")
#     save(save_path, fig)
#     println("Error plots saved to: $save_path")
    
#     return fig
# end

# 4. Plot residuals using Makie
function plot_residuals(dataTrue, fitData, savedir)
    """
    Creates residual plots (data - fit) using Makie.
    """
    x_data = dataTrue["x"]
    kD_data = dataTrue["KD"]
    output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]

    unique_kDs = unique(kD_data)
    println("Found $(length(unique_kDs)) unique kD values: $unique_kDs")

    # Filter out zero values from x_data for log scale
    valid_global_idx = x_data .> 0
    x_nonzero = x_data[valid_global_idx]
    println("\nGlobal x range (non-zero): $(minimum(x_nonzero)) to $(maximum(x_nonzero))")
    println("log10(x) range: $(log10(minimum(x_nonzero))) to $(log10(maximum(x_nonzero)))")
    
    
    
    # Create a figure for each unique kD
    figures = []

    for (kD_idx, kD_value) in enumerate(unique_kDs)
        # Find indices for this kD value
        kD_indices = findall(kD_data .== kD_value)

         # Filter indices to only include non-zero x values
        valid_kD_idx = [i for i in kD_indices if x_data[i] > 0]
        
        
        # Create figure with 4x2 grid for this kD
        fig = Figure(size=(1400, 1800))
    
        for (output_idx, output_name) in enumerate(output_names)
            row = ((output_idx-1) ÷ 2) + 1
            col = ((output_idx-1) % 2) + 1    
        
            # Extract data and fit
            data = dataTrue[output_name][valid_kD_idx]
            fit = dataTrue[output_name][valid_kD_idx]
            x_subset = x_data[valid_kD_idx]
        
            # Calculate residuals
            residuals = data .- fit
            mean_residual = mean(residuals)
        
            # Create axis
            ax = Makie.Axis(fig[row, col],
                  xlabel=(row == 4 ? "x" : ""),
                  ylabel="Data - Fit",
                  title="Residuals: kD = $kD_value: $output_name",
                  xscale =log10
                  )
                  
        
            # Plot residuals
            scatter!(ax, x_subset, residuals,
                 markersize=6,
                 color=:purple,
                 label="Residuals",
                 marker=:diamond)
        
            # Add zero line
            hlines!(ax, [0.0],
                color=:black,
                linewidth=1,
                linestyle=:dash,
                alpha=0.5,
                label="Zero Line")
        
            # Add mean residual line
            hlines!(ax, [mean_residual],
                color=:red,
                linewidth=2,
                label="Mean Residual = $(round(mean_residual, digits=5))")
        
            # Add legend to first plot only
            if output_idx == 1
                axislegend(ax, position=:rt)
            end
        
            # Add grid
            ax.xgridvisible = true
            ax.ygridvisible = true
        end
        # Add overall title for this figure
        Label(fig[0, :], "Residual Plots (All Outputs) - kD = $kD_value", 
              fontsize=20, font=:bold)
        
        # Adjust layout
        colgap!(fig.layout, 20)
        rowgap!(fig.layout, 20)
        
        # Save this figure
        save_path = joinpath(savedir, "residual_plots_kD_$(kD_value).png")
        save(save_path, fig)
        println("Saved fit vs data plot for kD = $kD_value to: $save_path")
        
        push!(figures, fig)
    end
    
    return figures
end

# Compute metrics function
# Compute metrics function per ligand condition
function compute_metrics_per_ligand_condition(dataTrue, fitData, savedir)
    """
    Computes RMSE and bias for each ligand condition (averaging O1 and O2).
    """
    output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    ligand_conds = ["00", "10", "01", "11"]
    metrics_dict = Dict{String, Float64}()
    
    println("Model Metrics (Per Ligand Condition):")
    println("-"^60)
    
    n = length(dataTrue["x"])
    
    for cond in ligand_conds
        # Get O1 and O2 data for this condition
        o1_data = dataTrue["O1_$cond"]
        o1_data_normed = o1_data ./ maximum(o1_data) # 
        o2_data = dataTrue["O2_$cond"]
        o2_data_normed = o2_data ./ maximum(o2_data)
        o1_fit = fitData["O1_$cond"]
        o1_fit_normed = o1_fit ./ maximum(o1_fit)
        o2_fit = fitData["O2_$cond"]
        o2_fit_normed = o1_fit ./ maximum(o2_fit)
        
        # Combine O1 and O2 for condition-level metrics
        all_data = vcat(o1_data, o2_data)
        all_fit = vcat(o1_fit, o2_fit)

        # standards
        # all_data_normed = vcat(o1_data_normed, o2_data_normed)
        # all_fit_normed = vcat(o1_fit_normed, o2_fit_normed)
        
        # Calculate RMSE for this condition
        rmse = sqrt(mean((all_data .- all_fit).^2))
        metrics_dict["RMSE_$cond"] = rmse

        # # standard RMSE for this condition
        # rmse = sqrt(mean((all_data_normed .- all_fit_normed).^2))
        # metrics_dict["StandardRMSE_$cond"] = rmse
        
        # Calculate bias for this condition
        bias = abs(sum(all_fit .> all_data)/length(all_data) - 0.5)
        metrics_dict["bias_$cond"] = bias
        
        # Also calculate individual RMSEs for reference
        rmse_o1 = sqrt(mean((o1_data .- o1_fit).^2))
        rmse_o2 = sqrt(mean((o2_data .- o2_fit).^2))
        metrics_dict["RMSE_O1_$cond"] = rmse_o1
        metrics_dict["RMSE_O2_$cond"] = rmse_o2
        
        # Print formatted output
        println("Ligand Condition $cond:")
        println("  Combined RMSE:  $(round(rmse, digits=6))")
        println("  O1 RMSE:                $(round(rmse_o1, digits=6))")
        println("  O2 RMSE:                $(round(rmse_o2, digits=6))")
        println("  Bias:                   $(round(bias, digits=6))")
        println()
    end
    
    # Calculate overall metrics across all conditions
    cond_rmse = [metrics_dict["RMSE_$cond"] for cond in ligand_conds]
    cond_bias = [metrics_dict["bias_$cond"] for cond in ligand_conds]
    
    metrics_dict["Worst_RMSE_all_conds"] = maximum(cond_rmse)
    metrics_dict["Worse_bias_all_conds"] = maximum(cond_bias)
    
    
    println("-"^60)
    println("Overall Metrics:")
    println("  Worse RMSE across conditions:  $(round(maximum(cond_rmse), digits=6))")
    println("  worst Bias across conditions:  $(round(maximum(cond_bias), digits=6))")
    
    # Save metrics
    metrics_path = joinpath(savedir, "model_metrics_per_condition.jld2")
    @save metrics_path metrics_dict
    println("\nMetrics saved to: $metrics_path")
    
    return metrics_dict
end

# Main function with Makie plots
function generate_all_plots_and_metrics(dataTrue, fitted_params, loss_history, savedir, model)
    """
    Convenience function to generate all plots and compute metrics using Makie.
    """
    println("\n" * "="^60)
    println("Generating Plots and Metrics")
    println("="^60)
    
    # Step 1: Generate fit data ONCE
    println("\n1. Generating model predictions...")
    fitData = generate_fit_data(dataTrue, fitted_params, model)
    
    # Step 2: Plot loss history with Makie
    println("\n2. Plotting loss history...")
    plot_loss_history(loss_history, savedir)
    
    # Step 3: Plot fit vs data with Makie
    println("\n3. Plotting fit vs data...")
    plot_fit_vs_data(dataTrue, fitData, savedir)
    
    # # Step 4: Plot error with Makie
    # println("\n4. Plotting error...")
    # plot_error_all_outputs(fakeData, fitData, savedir)
    
    # Step 5: Plot residuals with Makie
    println("\n5. Plotting residuals...")
    plot_residuals(dataTrue, fitData, savedir)
    
    # Step 6: Compute metrics
    println("\n6. Computing metrics...")
    metrics = compute_metrics_per_ligand_condition(dataTrue, fitData, savedir)
    
    println("\n" * "="^60)
    println("All plots and metrics generated successfully!")
    println("="^60)
    
    return metrics, fitData
end