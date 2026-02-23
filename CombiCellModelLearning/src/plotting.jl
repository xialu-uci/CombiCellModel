using ColorTypes
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
    Creates a single figure with 4x2 subplots comparing fitted model predictions to data,
    overlaying all kD values using color families (data=circles, fit=lines, color=kD value).
    """
    x_data = dataTrue["x"]
    kD_data = dataTrue["KD"]
    output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]

    # Get unique kD values
    unique_kDs = sort(unique(kD_data))
    n_kDs = length(unique_kDs)
    println("Found $n_kDs unique kD values: $unique_kDs")

    # Assign a color family per kD — use distinguishable colormap
    # Each kD gets a base color; data points are lighter, fit lines are darker
    base_colors = cgrad(:tab10, n_kDs, categorical=true)
    data_colors = [RGBAf(red(base_colors[k]), green(base_colors[k]), blue(base_colors[k]), 0.5) for k in 1:n_kDs]
    fit_colors  = [RGBAf(red(base_colors[k]), green(base_colors[k]), blue(base_colors[k]), 1.0) for k in 1:n_kDs]

    # Single figure with 4x2 grid
    fig = Figure(size=(1400, 1800))

    for (output_idx, output_name) in enumerate(output_names)
        row = ((output_idx - 1) ÷ 2) + 1
        col = ((output_idx - 1) % 2) + 1

        ax = Makie.Axis(fig[row, col],
                  xlabel=(row == 4 ? "x" : ""),
                  ylabel=output_name,
                  title=output_name,
                  xscale=log10)
        ax.xgridvisible = true
        ax.ygridvisible = true
        ax.xminorgridvisible = true

        for (k, kD_value) in enumerate(unique_kDs)
            # Get indices for this kD with nonzero x
            kD_indices = findall((kD_data .== kD_value) .& (x_data .> 0))

            x_subset = x_data[kD_indices]
            data_subset = dataTrue[output_name][kD_indices]
            fit_subset  = fitData[output_name][kD_indices]

            # Sort by x for smooth lines
            sort_idx = sortperm(x_subset)
            x_sorted   = x_subset[sort_idx]
            fit_sorted = fit_subset[sort_idx]

            # Plot data points
            scatter!(ax, x_subset, data_subset,
                     markersize=7,
                     color=data_colors[k],
                     marker=:circle,
                     label=(output_idx == 1 ? "kD=$kD_value data" : nothing))

            # Plot fit line
            lines!(ax, x_sorted, fit_sorted,
                   linewidth=2.5,
                   color=fit_colors[k],
                   label=(output_idx == 1 ? "kD=$kD_value fit" : nothing))
        end

        # Add legend only to first subplot
        if output_idx == 1
            axislegend(ax, position=:lt, framevisible=true, labelsize=10)
        end
    end

    Label(fig[0, :], "Model Fit vs Data (all kD values)", fontsize=20, font=:bold)
    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 20)

    save_path = joinpath(savedir, "fit_vs_data_overlay.png")
    save(save_path, fig)
    println("Saved overlaid fit vs data plot to: $save_path")

    return fig
end





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
    
    # println("Model Metrics (Per Ligand Condition):")
    # println("-"^60)
    
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
        # println("Ligand Condition $cond:")
        # println("  Combined RMSE:  $(round(rmse, digits=6))")
        # println("  O1 RMSE:                $(round(rmse_o1, digits=6))")
        # println("  O2 RMSE:                $(round(rmse_o2, digits=6))")
        # println("  Bias:                   $(round(bias, digits=6))")
        # println()
    end
    
    # Calculate overall metrics across all conditions
    cond_rmse = [metrics_dict["RMSE_$cond"] for cond in ligand_conds]
    cond_bias = [metrics_dict["bias_$cond"] for cond in ligand_conds]
    
    metrics_dict["Worst_RMSE_all_conds"] = maximum(cond_rmse)
    metrics_dict["Worse_bias_all_conds"] = maximum(cond_bias)
    
    
    # println("-"^60)
    # println("Overall Metrics:")
    # println("  Worse RMSE across conditions:  $(round(maximum(cond_rmse), digits=6))")
    # println("  worst Bias across conditions:  $(round(maximum(cond_bias), digits=6))")
    
    # Save metrics
    metrics_path = joinpath(savedir, "model_metrics_per_condition.jld2")
    @save metrics_path metrics_dict
    println("\nMetrics saved to: $metrics_path")
    
    return metrics_dict
end

function compute_param_ratios(i,j, fitted_params)

#     intPoints = ["fI", "alpha", "tT", "g1", "k_on_2d", "kP", "nKP", "lamdaX", "nC", "XO1", "O1max", "O2max"] # 13 = extraCD2, 14 = extraPD1
#     string_to_idx = Dict(label => i for (i, label) in enumerate(intPoints))
#    #  metrics_dict = Dict{String, Float64}()

#     parts = split(folder, "-")
#         if length(parts) == 4 && parts[1] == "cd2" && parts[3] == "pd1"
#             i_str = parts[2]
#             j_str = parts[4]

#             # Check if strings are valid
#             if haskey(string_to_idx, i_str) && haskey(string_to_idx, j_str)
#                 i = string_to_idx[i_str]
#                 j = string_to_idx[j_str]
#             else
#                 @warn "Unknown label in folder name: $folder (i_str='$i_str', j_str='$j_str')"
#             end
#         end

    cd2_ratio = fitted_params[13]/fitted_params[i]
    pd1_ratio = fitted_params[14]/fitted_params[j]
    return cd2_ratio, pd1_ratio

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

using JLD2, CairoMakie

function create_metrics_heatmaps(base_path::String)
    """
    Creates 12x12 heatmaps for worst RMSE, worst bias, cd2 ratio, and pd1 ratio from all model folders.
    
    Args:
        base_path: Path to the parent folder containing all model folders (e.g., "02182026_fakeData")
    
    Returns:
        Dictionary with heatmap figures
    """
    
    # Define the intPoints order
    intPoints = ["fI", "alpha", "tT", "g1", "k_on_2d", "kP", "nKP", "lamdaX", "nC", "XO1", "O1max", "O2max"]
    string_to_idx = Dict(label => i for (i, label) in enumerate(intPoints))
    
    # Get all folders matching pattern
    all_folders = readdir(base_path)
    model_folders = filter(f -> occursin(r"^cd2-.+-pd1-.+$", f), all_folders)
    
    println("Found $(length(model_folders)) model folders")
    
    # Parse intPoints from folder names
    # Format: cd2-{i}-pd1-{j}
    rmse_matrix = fill(NaN, 12, 12)
    bias_matrix = fill(NaN, 12, 12)
    cd2_ratio_matrix = fill(NaN, 12, 12)
    pd1_ratio_matrix = fill(NaN, 12, 12)

    
    # Track which indices we've found
    i_vals = Int[]
    j_vals = Int[]
    
    for folder in model_folders
        # Parse folder name
        parts = split(folder, "-")
        if length(parts) == 4 && parts[1] == "cd2" && parts[3] == "pd1"
            i_str = parts[2]
            j_str = parts[4]

            # Check if strings are valid
            if haskey(string_to_idx, i_str) && haskey(string_to_idx, j_str)
                i = string_to_idx[i_str]
                j = string_to_idx[j_str]
                
                # Load metrics file
                metrics_path = joinpath(base_path, folder, "model_metrics_per_condition.jld2")
                params_path = joinpath(base_path,folder, "final_params_derepr.jld2") 
                

                if isfile(metrics_path)
                    metrics = load(metrics_path)["metrics_dict"]
                    params = load(params_path)["final_params_derepr"]
                    p_class = params.p_classical

                    # Extract worst metrics
                    if haskey(metrics, "Worst_RMSE_all_conds")
                        val = metrics["Worst_RMSE_all_conds"]
                        if !isnan(val) && !isinf(val)
                            rmse_matrix[i, j] = val
                        end
                    end
                    
                    # Handle bias with both possible keys
                    if haskey(metrics, "Worse_bias_all_conds")
                        val = metrics["Worse_bias_all_conds"]
                        if !isnan(val) && !isinf(val)
                            bias_matrix[i, j] = val
                        end
                    elseif haskey(metrics, "Worst_bias_all_conds")
                        val = metrics["Worst_bias_all_conds"]
                        if !isnan(val) && !isinf(val)
                            bias_matrix[i, j] = val
                        end
                    end

                    cd2_ratio, pd1_ratio = compute_param_ratios(i, j, p_class)
                    cd2_ratio_matrix[i, j] = cd2_ratio
                    pd1_ratio_matrix[i, j] = pd1_ratio
                    
                    push!(i_vals, i)
                    push!(j_vals, j)
                else
                    @warn "No metrics file found in $folder"
                end
            else
                @warn "Unknown label in folder name: $folder (i_str='$i_str', j_str='$j_str')"
            end
        end
    end
    
    # Check what range of indices we have
    i_range = sort(unique(i_vals))
    j_range = sort(unique(j_vals))
    println("\nFound indices:")
    println("  i values (cd2-{i}): $i_range")
    println("  j values (pd1-{j}): $j_range")
    
    # Get valid ranges for color scaling
    valid_rmse = filter(!isnan, rmse_matrix[:])
    valid_rmse = filter(!isinf, valid_rmse)
    valid_bias = filter(!isnan, bias_matrix[:])
    valid_bias = filter(!isinf, valid_bias)
    valid_cd2_ratio = filter(!isnan, cd2_ratio_matrix[:])
    valid_cd2_ratio = filter(!isinf, valid_cd2_ratio)
    valid_pd1_ratio = filter(!isnan, pd1_ratio_matrix[:])
    valid_pd1_ratio = filter(!isinf, valid_pd1_ratio)
    
    if isempty(valid_rmse)
        error("No valid RMSE data found to plot!")
    end
    
    if isempty(valid_bias)
        error("No valid bias data found to plot!")
    end
    
    # Create heatmaps
    figures = Dict{String, Figure}()
    
    # Helper function to set up a standard axis
    function make_axis(fig, title, xlabel, ylabel)
        ax = Makie.Axis(fig[1, 1],
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=(1:12, intPoints),
                  yticks=(1:12, intPoints))
        ax.xticklabelrotation = π/4
        ax.xticklabelalign = (:right, :center)
        return ax
    end

    # Helper to annotate heatmap cells
    function annotate_cells!(ax, matrix; digits=3)
        for i in 1:12, j in 1:12
            if !isnan(matrix[i, j]) && !isinf(matrix[i, j])
                text!(ax, i, j,
                      text=string(round(matrix[i, j], digits=digits)),
                      color=:white,
                      align=(:center, :center),
                      fontsize=8,
                      strokewidth=0.5,
                      strokecolor=:black)
            end
        end
    end

    # RMSE Heatmap
    fig_rmse = Figure(size=(1200, 1000))
    ax_rmse = make_axis(fig_rmse, "Worst RMSE Across All Conditions", "cd2 parameter", "pd1 parameter")
    hm_rmse = heatmap!(ax_rmse, 1:12, 1:12, rmse_matrix,
                       colormap=:viridis,
                       colorrange=(minimum(valid_rmse), maximum(valid_rmse)),
                       nan_color=:lightgray)
    annotate_cells!(ax_rmse, rmse_matrix)
    Colorbar(fig_rmse[1, 2], hm_rmse, label="RMSE")
    
    # Bias Heatmap
    fig_bias = Figure(size=(1200, 1000))
    ax_bias = make_axis(fig_bias, "Worst Bias Across All Conditions", "cd2 parameter", "pd1 parameter")
    hm_bias = heatmap!(ax_bias, 1:12, 1:12, bias_matrix,
                       colormap=:plasma,
                       colorrange=(minimum(valid_bias), maximum(valid_bias)),
                       nan_color=:lightgray)
    annotate_cells!(ax_bias, bias_matrix)
    Colorbar(fig_bias[1, 2], hm_bias, label="Bias")

    # CD2 Ratio Heatmap
    fig_cd2 = Figure(size=(1200, 1000))
    ax_cd2 = make_axis(fig_cd2, "CD2 Parameter Ratio", "cd2 parameter", "pd1 parameter")
    hm_cd2 = heatmap!(ax_cd2, 1:12, 1:12, cd2_ratio_matrix,
                      colormap=:RdBu,
                      colorrange=isempty(valid_cd2_ratio) ? (0, 1) : (minimum(valid_cd2_ratio), maximum(valid_cd2_ratio)),
                      nan_color=:lightgray)
    annotate_cells!(ax_cd2, cd2_ratio_matrix)
    Colorbar(fig_cd2[1, 2], hm_cd2, label="CD2 Ratio")

    # PD1 Ratio Heatmap
    fig_pd1 = Figure(size=(1200, 1000))
    ax_pd1 = make_axis(fig_pd1, "PD1 Parameter Ratio", "cd2 parameter", "pd1 parameter")
    hm_pd1 = heatmap!(ax_pd1, 1:12, 1:12, pd1_ratio_matrix,
                      colormap=:RdBu,
                      colorrange=isempty(valid_pd1_ratio) ? (0, 1) : (minimum(valid_pd1_ratio), maximum(valid_pd1_ratio)),
                      nan_color=:lightgray)
    annotate_cells!(ax_pd1, pd1_ratio_matrix)
    Colorbar(fig_pd1[1, 2], hm_pd1, label="PD1 Ratio")
    
    # Save figures
    save(joinpath(base_path, "rmse_heatmap.png"), fig_rmse)
    save(joinpath(base_path, "bias_heatmap.png"), fig_bias)
    save(joinpath(base_path, "cd2_ratio_heatmap.png"), fig_cd2)
    save(joinpath(base_path, "pd1_ratio_heatmap.png"), fig_pd1)

    println("\nHeatmaps saved to:")
    println("  RMSE:      $(joinpath(base_path, "rmse_heatmap.png"))")
    println("  Bias:      $(joinpath(base_path, "bias_heatmap.png"))")
    println("  CD2 Ratio: $(joinpath(base_path, "cd2_ratio_heatmap.png"))")
    println("  PD1 Ratio: $(joinpath(base_path, "pd1_ratio_heatmap.png"))")
    
    figures["rmse"] = fig_rmse
    figures["bias"] = fig_bias
    figures["cd2_ratio"] = fig_cd2
    figures["pd1_ratio"] = fig_pd1
    
    # Print summary statistics
    println("\n" * "="^60)
    println("Summary Statistics:")
    println("="^60)
    println("RMSE      - Min: $(minimum(valid_rmse)), Max: $(maximum(valid_rmse)), Mean: $(mean(valid_rmse))")
    println("Bias      - Min: $(minimum(valid_bias)), Max: $(maximum(valid_bias)), Mean: $(mean(valid_bias))")
    if !isempty(valid_cd2_ratio)
        println("CD2 Ratio - Min: $(minimum(valid_cd2_ratio)), Max: $(maximum(valid_cd2_ratio)), Mean: $(mean(valid_cd2_ratio))")
    end
    if !isempty(valid_pd1_ratio)
        println("PD1 Ratio - Min: $(minimum(valid_pd1_ratio)), Max: $(maximum(valid_pd1_ratio)), Mean: $(mean(valid_pd1_ratio))")
    end
    println("\nCells with valid RMSE: $(length(valid_rmse))/144")
    println("Cells with valid Bias: $(length(valid_bias))/144")
    
    return figures, rmse_matrix, bias_matrix, cd2_ratio_matrix, pd1_ratio_matrix
end