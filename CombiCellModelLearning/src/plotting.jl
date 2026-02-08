# function generate_fit_data(fakeData, p_class, model)
#     """
#     Generates model predictions for all outputs using fitted parameters.
#     Returns a dictionary with the same structure as fakeData but with fitted values.
    
#     Args:
#         fakeData: Dictionary containing input data (x, KD)
#         fitted_params: Fitted parameters from the model
        
#     Returns:
#         fitData: Dictionary containing fitted values for all outputs
#     """
#     # Extract input data
#     x_data = fakeData["x"]
#     kD_data = fakeData["KD"]
#     # p_class = fitted_params.p_classical

#     predictions = CombiCellModelLearning.forward_combi(x_data, kD_data, p_class, model)
    
#     O1_00_fit = predictions[:, 1]
#     O2_00_fit = predictions[:, 2]
#     O1_10_fit = predictions[:, 3]
#     O2_10_fit = predictions[:, 4]
#     O1_01_fit = predictions[:, 5]
#     O2_01_fit = predictions[:, 6]
#     O1_11_fit = predictions[:, 7]
#     O2_11_fit = predictions[:, 8]
    
#     # Get predictions for all 8 outputs
    
    
#     # Create fitData dictionary with same structure as fakeData
#     fitData = Dict{String, Vector{Float64}}(
#         "x" => x_data,  # Include inputs for convenience
#         "KD" => kD_data,
#         "O1_00" => O1_00_fit,
#         "O2_00" => O2_00_fit,
#         "O1_10" => O1_10_fit,
#         "O2_10" => O2_10_fit,
#         "O1_01" => O1_01_fit,
#         "O2_01" => O2_01_fit,
#         "O1_11" => O1_11_fit,
#         "O2_11" => O2_11_fit
#     )
    
#     return fitData
# end

# # 1. Plot loss history (unchanged, doesn't need fitData)
# function plot_loss_history(loss_history, savedir)
#     """
#     Plots the loss history from the optimization process.
    
#     Args:
#         loss_history: Array of loss values at each iteration
#         savedir: Directory to save the plot
#     """
#     p = plot(loss_history, 
#              xlabel="Iteration", 
#              ylabel="Loss", 
#              title="Loss History During Optimization",
#              label="Loss",
#              linewidth=2,
#              yscale=:log10,
#              legend=:topright,
#              grid=true,
#              color=:blue)
    
#     save_path = joinpath(savedir, "loss_history.png")
#     savefig(p, save_path)
#     println("Loss history plot saved to: $save_path")
    
#     return p
# end

# # 2. Plot fit vs data using pre-computed fitData
# function plot_fit_vs_data(fakeData, fitData, savedir)
#     """
#     Creates 8 plots comparing fitted model predictions to simulated data.
#     Uses pre-computed fitData to avoid redundant calculations.
    
#     Args:
#         fakeData: Dictionary containing simulated data
#         fitData: Dictionary containing fitted values
#         savedir: Directory to save the plots
#     """
#     # Get x data from either dictionary
#     x_data = fakeData["x"]
    
#     # Output names for all 8 outputs
#     output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    
#     # Create a 4x2 grid of plots
#     plots = []
    
#     for output_name in output_names
#         # Extract data and fit
#         data = fakeData[output_name]
#         fit = fitData[output_name]
        
#         # Create individual plot
#         p = scatter(x_data, data, 
#                     label="Data",
#                     markersize=5,
#                     marker=:circle,
#                     alpha=0.7,
#                     xlabel="x",
#                     ylabel=output_name,
#                     title="Fit vs Data: $output_name",
#                     legend=:topleft)
        
#         # Add fitted curve
#         plot!(p, x_data, fit,
#               label="Fit",
#               linewidth=2,
#               color=:red)
        
#         push!(plots, p)
#     end
    
#     # Combine all plots
#     combined_plot = plot(plots..., 
#                          layout=(4, 2),
#                          size=(1200, 1600),
#                          plot_title="Model Fit vs Simulated Data (All Outputs)")
    
#     save_path = joinpath(savedir, "fit_vs_data_all_outputs.png")
#     savefig(combined_plot, save_path)
#     println("Fit vs data plots saved to: $save_path")
    
#     return combined_plot
# end

# # 3. Plot error using pre-computed fitData
# function plot_error_all_outputs(fakeData, fitData, savedir)
#     """
#     Creates 8 error plots showing absolute differences between data and fit.
#     Uses pre-computed fitData to avoid redundant calculations.
    
#     Args:
#         fakeData: Dictionary containing simulated data
#         fitData: Dictionary containing fitted values
#         savedir: Directory to save the plots
#     """
#     x_data = fakeData["x"]
#     output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    
#     plots = []
    
#     for output_name in output_names
#         # Extract data and fit
#         data = fakeData[output_name]
#         fit = fitData[output_name]
        
#         # Calculate absolute error
#         error = abs.(data .- fit)
        
#         # Create individual error plot
#         p = scatter(x_data, error,
#                     label="Absolute Error",
#                     markersize=5,
#                     marker=:diamond,
#                     color=:red,
#                     xlabel="x",
#                     ylabel="|Data - Fit|",
#                     title="Error: $output_name",
#                     legend=:topright)
        
#         # Add mean error line
#         mean_error = mean(error)
#         hline!(p, [mean_error],
#                label="Mean Error = $(round(mean_error, digits=5))",
#                linewidth=2,
#                linestyle=:dash,
#                color=:blue)
        
#         push!(plots, p)
#     end
    
#     # Combine all plots
#     combined_plot = plot(plots..., 
#                          layout=(4, 2),
#                          size=(1200, 1600),
#                          plot_title="Error Plots (All Outputs)")
    
#     save_path = joinpath(savedir, "error_plots_all_outputs.png")
#     savefig(combined_plot, save_path)
#     println("Error plots saved to: $save_path")
    
#     return combined_plot
# end

# # 4. Compute metrics using pre-computed fitData
# function compute_metrics_all_outputs(fakeData, fitData, savedir)
#     """
#     Computes RMSE for each of the 8 output variables.
#     Uses pre-computed fitData to avoid redundant calculations.
    
#     Args:
#         fakeData: Dictionary containing simulated data
#         fitData: Dictionary containing fitted values
#         savedir: Directory to save the metrics
        
#     Returns:
#         metrics_dict: Dictionary containing RMSE for each output
#     """
#     output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
#     metrics_dict = Dict{String, Float64}()
    
#     println("RMSE Metrics:")
#     println("-"^40)
    
#     # bias = 0
#     n = length(fakeData["x"])
#     for output_name in output_names
#         # Extract data and fit
#         data = fakeData[output_name]
#         fit = fitData[output_name]
        
#         # Calculate RMSE
#         rmse = sqrt(mean((data .- fit).^2))
#         metrics_dict["RMSE_$output_name"] = rmse

#         # Calculate bias 

#         bias = abs(sum(fit .> data)/n - 0.5)
#         metrics_dict["bias_$output_name"] = bias
        
#         # Print formatted output
#         println("RMSE_$output_name: $(lpad(round(rmse, digits=6), 12))")
#         println("bias_$output_name: $(lpad(round(bias, digits=6), 12))")
#     end
   
    
#     # Calculate average RMSE
#     # avg_rmse = mean(values(metrics_dict))
#     # metrics_dict["Average_RMSE"] = avg_rmse
    
#     # println("-"^40)
#     # println("Average RMSE: $(lpad(round(avg_rmse, digits=6), 16))")
    
#     # Save metrics
#     metrics_path = joinpath(savedir, "model_metrics.jld2")
#     @save metrics_path metrics_dict
#     println("\nMetrics saved to: $metrics_path")
    
#     return metrics_dict
# end

# # 5. Additional utility function: Plot residuals (alternative to absolute error)
# function plot_residuals(fakeData, fitData, savedir)
#     """
#     Creates residual plots (data - fit) instead of absolute error.
#     Residuals can show systematic patterns in errors.
    
#     Args:
#         fakeData: Dictionary containing simulated data
#         fitData: Dictionary containing fitted values
#         savedir: Directory to save the plots
#     """
#     x_data = fakeData["x"]
#     output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    
#     plots = []
    
#     for output_name in output_names
#         # Extract data and fit
#         data = fakeData[output_name]
#         fit = fitData[output_name]
        
#         # Calculate residuals (data - fit)
#         residuals = data .- fit
        
#         # Create residual plot
#         p = scatter(x_data, residuals,
#                     label="Residuals",
#                     markersize=5,
#                     marker=:circle,
#                     color=:purple,
#                     xlabel="x",
#                     ylabel="Data - Fit",
#                     title="Residuals: $output_name",
#                     legend=:topright)
        
#         # Add horizontal line at y=0 for reference
#         hline!(p, [0.0],
#                label="Zero Line",
#                linewidth=1,
#                linestyle=:dash,
#                color=:black,
#                alpha=0.5)
        
#         # Add mean residual line
#         mean_residual = mean(residuals)
#         hline!(p, [mean_residual],
#                label="Mean Residual = $(round(mean_residual, digits=5))",
#                linewidth=2,
#                linestyle=:solid,
#                color=:red)
        
#         push!(plots, p)
#     end
    
#     # Combine all plots
#     combined_plot = plot(plots..., 
#                          layout=(4, 2),
#                          size=(1200, 1600),
#                          plot_title="Residual Plots (All Outputs)")
    
#     save_path = joinpath(savedir, "residual_plots_all_outputs.png")
#     savefig(combined_plot, save_path)
#     println("Residual plots saved to: $save_path")
    
#     return combined_plot
# end

# # Main function to run all plots and metrics
# function generate_all_plots_and_metrics(fakeData, fitted_params, loss_history, savedir, model)
#     """
#     Convenience function to generate all plots and compute metrics.
#     Efficiently computes fit data once and reuses it.
    
#     Args:
#         fakeData: Dictionary containing simulated data
#         fitted_params: Parameters from the fitted model
#         loss_history: Array of loss values
#         savedir: Directory to save all outputs
#     """
#     println("\n" * "="^50)
#     println("Generating Plots and Metrics")
#     println("="^50)
    
#     # Step 1: Generate fit data ONCE
#     println("\n1. Generating model predictions...")
#     fitData = generate_fit_data(fakeData, fitted_params, model)
    
#     # Step 2: Plot loss history
#     println("\n2. Plotting loss history...")
#     plot_loss_history(loss_history, savedir)
    
#     # Step 3: Plot fit vs data
#     println("\n3. Plotting fit vs data...")
#     plot_fit_vs_data(fakeData, fitData, savedir)
    
#     # # Step 4: Plot error
#     # println("\n4. Plotting error...")
#     # plot_error_all_outputs(fakeData, fitData, savedir)
    
#     # Step 5: Plot residuals
#     println("\n5. Plotting residuals...")
#     plot_residuals(fakeData, fitData, savedir)
    
#     # Step 6: Compute metrics
#     println("\n6. Computing metrics...")
#     metrics = compute_metrics_all_outputs(fakeData, fitData, savedir)
    
#     println("\n" * "="^50)
#     println("All plots and metrics generated successfully!")
#     println("="^50)
    
#     return metrics, fitData
# end


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
    fig = Figure(resolution=(800, 600))
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
function plot_fit_vs_data(fakeData, fitData, savedir)
    """
    Creates 8 plots comparing fitted model predictions to simulated data using Makie.
    """
    x_data = fakeData["x"]
    output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    
    # Create figure with 4x2 grid
    fig = Figure(resolution=(1400, 1800))
    
    for (idx, output_name) in enumerate(output_names)
        # Calculate row and column
        row = ((idx-1) ÷ 2) + 1
        col = ((idx-1) % 2) + 1
        
        # Extract data and fit
        data = fakeData[output_name]
        fit = fitData[output_name]
        
        # should fix weird polygons in my fit
        sort_idx = sortperm(x_data)
        x_sorted = x_data[sort_idx]
        fit_sorted = fit[sort_idx]
        
        # Create axis for this subplot
        ax = Makie.Axis(fig[row, col],
                  xlabel=(row == 4 ? "x" : ""),
                  ylabel=output_name,
                  title="Fit vs Data: $output_name")
        
        # Plot data points
        scatter!(ax, x_data, data,
                 markersize=6,
                 color=(:steelblue, 0.7),
                 label="Data",
                 marker=:circle)
        
        # Plot fitted curve
        lines!(ax, x_sorted, fit_sorted,
               linewidth=2.5,
               color=:red,
               label="Fit")
        
        # Add legend to first plot only to save space
        if idx == 1
            axislegend(ax, position=:lt)
        end
        
        # Add grid
        ax.xgridvisible = true
        ax.ygridvisible = true
    end
    
    # Add overall title
    Label(fig[0, :], "Model Fit vs Simulated Data (All Outputs)", 
          fontsize=20, font=:bold)
    
    # Adjust layout
    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 20)
    
    # Save figure
    save_path = joinpath(savedir, "fit_vs_data_all_outputs_makie.png")
    save(save_path, fig)
    println("Fit vs data plots saved to: $save_path")
    
    return fig
end

# 3. Plot error using Makie
function plot_error_all_outputs(fakeData, fitData, savedir)
    """
    Creates 8 error plots showing absolute differences between data and fit using Makie.
    """
    x_data = fakeData["x"]
    output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    
    # Create figure with 4x2 grid
    fig = Figure(resolution=(1400, 1800))
    
    for (idx, output_name) in enumerate(output_names)
        row = ((idx-1) ÷ 2) + 1
        col = ((idx-1) % 2) + 1
        
        # Extract data and fit
        data = fakeData[output_name]
        fit = fitData[output_name]
        
        # Calculate absolute error
        error = abs.(data .- fit)
        mean_error = mean(error)
        
        # Create axis
        ax = Makie.Axis(fig[row, col],
                  xlabel=(row == 4 ? "x" : ""),
                  ylabel="|Data - Fit|",
                  title="Error: $output_name")
        
        # Plot error points
        scatter!(ax, x_data, error,
                 markersize=6,
                 color=:red,
                 label="Absolute Error",
                 marker=:diamond)
        
        # Add mean error line
        hlines!(ax, [mean_error],
                color=:blue,
                linewidth=2,
                linestyle=:dash,
                label="Mean Error = $(round(mean_error, digits=5))")
        
        # Add legend to first plot only
        if idx == 1
            axislegend(ax, position=:rt)
        end
        
        # Add grid
        ax.xgridvisible = true
        ax.ygridvisible = true
    end
    
    # Add overall title
    Label(fig[0, :], "Error Plots (All Outputs)", 
          fontsize=20, font=:bold)
    
    # Adjust layout
    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 20)
    
    save_path = joinpath(savedir, "error_plots_all_outputs_makie.png")
    save(save_path, fig)
    println("Error plots saved to: $save_path")
    
    return fig
end

# 4. Plot residuals using Makie
function plot_residuals(fakeData, fitData, savedir)
    """
    Creates residual plots (data - fit) using Makie.
    """
    x_data = fakeData["x"]
    output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    
    # Create figure with 4x2 grid
    fig = Figure(resolution=(1400, 1800))
    
    for (idx, output_name) in enumerate(output_names)
        row = ((idx-1) ÷ 2) + 1
        col = ((idx-1) % 2) + 1
        
        # Extract data and fit
        data = fakeData[output_name]
        fit = fitData[output_name]
        
        # Calculate residuals
        residuals = data .- fit
        mean_residual = mean(residuals)
        
        # Create axis
        ax = Makie.Axis(fig[row, col],
                  xlabel=(row == 4 ? "x" : ""),
                  ylabel="Data - Fit",
                  title="Residuals: $output_name")
        
        # Plot residuals
        scatter!(ax, x_data, residuals,
                 markersize=6,
                 color=:purple,
                 label="Residuals",
                 marker=:circle)
        
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
        if idx == 1
            axislegend(ax, position=:rt)
        end
        
        # Add grid
        ax.xgridvisible = true
        ax.ygridvisible = true
    end
    
    # Add overall title
    Label(fig[0, :], "Residual Plots (All Outputs)", 
          fontsize=20, font=:bold)
    
    # Adjust layout
    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 20)
    
    save_path = joinpath(savedir, "residual_plots_all_outputs_makie.png")
    save(save_path, fig)
    println("Residual plots saved to: $save_path")
    
    return fig
end

# Compute metrics function
function compute_metrics_all_outputs(fakeData, fitData, savedir)
    """
    Computes RMSE and bias for each of the 8 output variables.
    """
    output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    metrics_dict = Dict{String, Float64}()
    
    println("Model Metrics:")
    println("-"^50)
    
    n = length(fakeData["x"])
    for output_name in output_names
        data = fakeData[output_name]
        fit = fitData[output_name]
        
        # Calculate RMSE
        rmse = sqrt(mean((data .- fit).^2))
        metrics_dict["RMSE_$output_name"] = rmse
        
        # Calculate bias
        bias = abs(sum(fit .> data)/n - 0.5)
        metrics_dict["bias_$output_name"] = bias
        
        # Print formatted output
        println("$output_name:")
        println("  RMSE:  $(round(rmse, digits=6))")
        println("  Bias:  $(round(bias, digits=6))")
        println()
    end
    
    # Calculate overall metrics
    all_rmse = [metrics_dict["RMSE_$name"] for name in output_names]
    all_bias = [metrics_dict["bias_$name"] for name in output_names]
    
    metrics_dict["Average_RMSE"] = mean(all_rmse)
    metrics_dict["Average_bias"] = mean(all_bias)
    
    println("-"^50)
    println("Overall Metrics:")
    println("  Average RMSE:  $(round(mean(all_rmse), digits=6))")
    println("  Average Bias:  $(round(mean(all_bias), digits=6))")
    
    # Save metrics
    metrics_path = joinpath(savedir, "model_metrics.jld2")
    @save metrics_path metrics_dict
    println("\nMetrics saved to: $metrics_path")
    
    return metrics_dict
end

# Main function with Makie plots
function generate_all_plots_and_metrics(fakeData, fitted_params, loss_history, savedir, model)
    """
    Convenience function to generate all plots and compute metrics using Makie.
    """
    println("\n" * "="^60)
    println("Generating Plots and Metrics (Using Makie)")
    println("="^60)
    
    # Step 1: Generate fit data ONCE
    println("\n1. Generating model predictions...")
    fitData = generate_fit_data(fakeData, fitted_params, model)
    
    # Step 2: Plot loss history with Makie
    println("\n2. Plotting loss history...")
    plot_loss_history(loss_history, savedir)
    
    # Step 3: Plot fit vs data with Makie
    println("\n3. Plotting fit vs data...")
    plot_fit_vs_data(fakeData, fitData, savedir)
    
    # Step 4: Plot error with Makie
    println("\n4. Plotting error...")
    plot_error_all_outputs(fakeData, fitData, savedir)
    
    # Step 5: Plot residuals with Makie
    println("\n5. Plotting residuals...")
    plot_residuals(fakeData, fitData, savedir)
    
    # Step 6: Compute metrics
    println("\n6. Computing metrics...")
    metrics = compute_metrics_all_outputs(fakeData, fitData, savedir)
    
    println("\n" * "="^60)
    println("All plots and metrics generated successfully!")
    println("="^60)
    
    return metrics, fitData
end