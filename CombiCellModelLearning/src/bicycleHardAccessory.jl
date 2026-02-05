
using CombiCellModelLearning
using ComponentArrays # i feel like I shouldn't need this in here...
using Optimization
using OptimizationBBO
using Statistics
using JLD2
using Plots

# using OptimizationBBO

include("sim_data.jl")

x_for_sim = [0.1, 0.02, 0.004, 0.0008, 0.00016, 0.000032, 0.0000064, 0.00000128, 0.000000256, 5.12e-8, 1.024e-8, 0.00, 0.1, 0.02, 0.004, 0.0008, 0.00016, 0.000032, 0.0000064, 0.00000128, 0.000000256, 5.12e-8, 1.024e-8, 0.00, 0.1, 0.02, 0.004, 0.0008, 0.00016, 0.000032, 0.0000064, 0.00000128, 0.000000256, 5.12e-8, 1.024e-8, 0.00]
kD_for_sim = [8.0 for _ in x_for_sim]
stdevs_for_sim = [0.005 for _ in x_for_sim]



params_for_sim = ComponentArray(
    fI=0.6,
    alpha=2e6,
    tT=500.0,
    g1=0.7,
    k_on_2d=15.0,
    kP=0.4,
    nKP=1.8,
    lambdaX=0.05,
    nC=2.1,
    XO1=0.4,
    O1max=0.95,
    O2max=110.0,
    extraCD2 = 0.99,
    extraPD1 = 65
)


# basically copied over from fw() in define_modelCombi_classical.jl

function true_fw(x::Vector{Float64}, kD::Vector{Float64}, params)
    # unpack params
    fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max, extraCD2, extraPD1 = params
    # hypothesis = CD2 affects O2max, PD1 affects O1max
    params_00 = ComponentArray(
        fI=fI,
        alpha=alpha,
        tT=tT,
        g1=g1,
        k_on_2d=k_on_2d,
        kP=kP,
        nKP=nKP,
        lambdaX=lambdaX,
        nC=nC,
        XO1=XO1,
        O1max=O1max,
        O2max=O2max,
    )

    O1_00, O2_00 = true_fw_inside(x, kD, params_00)

    params_10 = ComponentArray(
        fI=fI,
        alpha=alpha,
        tT=tT,
        g1=g1,
        k_on_2d=k_on_2d,
        kP=kP,
        nKP=nKP,
        lambdaX=lambdaX,
        nC=nC,
        XO1=XO1,
        O1max=extraCD2,
        O2max=O2max,
    )

    O1_10, O2_10 = true_fw_inside(x, kD, params_10)


    params_01 = ComponentArray(
        fI=fI,
        alpha=alpha,
        tT=tT,
        g1=g1,
        k_on_2d=k_on_2d,
        kP=kP,
        nKP=nKP,
        lambdaX=lambdaX,
        nC=nC,
        XO1=XO1,
        O1max=O1max,
        O2max=extraPD1,
    )

    O1_01, O2_01 = true_fw_inside(x, kD, params_01)

    params_11 = ComponentArray(
        fI=fI,
        alpha=alpha,
        tT=tT,
        g1=g1,
        k_on_2d=k_on_2d,
        kP=kP,
        nKP=nKP,
        lambdaX=lambdaX,
        nC=nC,
        XO1=XO1,
        O1_max=extraCD2,
        O2_max=extraPD1,
    )
    
    O1_11, O2_11 = true_fw_inside(x, kD, params_11)

    return O1_00, O2_00,O1_10, O2_10, O1_01, O2_01, O1_11, O2_11
    
    
end
function true_fw_inside(x::Vector{Float64}, kD::Vector{Float64}, params)
    # unpack params
    fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max = params

    O1 = Float64[]
    O2 = Float64[]

    for (xi, kDi) in zip(x, kD)
        CT = (1-fI) * (alpha * xi + tT + g1 * kDi / k_on_2d - sqrt((alpha * xi + tT + g1 * kDi / k_on_2d)^2 - 4 * alpha * xi * tT)) / 2
        CN = (1 / (1 + g1 * kDi / kP))^nKP * CT
        X = CN^nC / (lambdaX^nC + CN^nC)

        O1_val = XO1 / (XO1 + X)
        O2_val = X

        O1i = O1max * O1_val 
        O2i = O2max * O2_val

        push!(O1, O1i)
        push!(O2, O2i)
    end
    return (O1, O2)
end

fakeData = sim_data(x_for_sim, kD_for_sim, stdevs_for_sim, params_for_sim, true_fw)


# now let's make a classical model and try to fit parameters to the simulated data
# differential evolution


model = CombiCellModelLearning.make_ModelCombiClassic()
p_repr_ig = deepcopy(model.params_repr_ig)
# learning problem
learning_problem = CombiCellModelLearning.LearningProblem(
     data =fakeData,
     model= model,
     p_repr_lb=CombiCellModelLearning.represent(model.p_derepresented_lowerbounds, model),
     p_repr_ub=CombiCellModelLearning.represent(model.p_derepresented_upperbounds, model),
     mask = trues(length(x_for_sim)), # no mask for now
     loss_strategy="vanilla")

# for bbo, use solve(prob, algo, maxiters, callback)

# prob is an optimizationproblem
# OptimizationProblem(objective_function, initial_guess, p = either constant prams or fixed used in objective function?? = [1,100.0]???, lb, ub)

# for objective function, need to define function that takes in parameter array and returns loss (pass get_loss with reconstructed params)
function obj_func(x, p)
    p_repr = CombiCellModelLearning.reconstruct_learning_params_from_array(x, p_repr_ig, model) # this is where params are updated # the trick is the x is the actual params we want. 
    # only pass through p_repr_ig for the keys
    return CombiCellModelLearning.get_loss(p_repr; learning_problem=learning_problem)
end

# let's make this whole section a function bbo_learn(learning_problem, p_repr_ig)
function bbo_learn(learning_problem, p_repr_ig)
    # initial guess params array
    classical_params_array = collect(values(copy(p_repr_ig)))

    # p
    p = [1.0, 100.0] # not used in obj_func, honestly I think i could delete this but will leave in for now and see if it changes anything later

    # algo is the optimization algorithm (here bbo)
    # maxiters is maximum iterations
    maxiters = 30000 # reduced for testing
    # callback is a function called at each iteration, s.t. optimzation stops if it returns true
    config = CallbackConfig() # just stores info for callback function in fields
    callback, loss_history = CombiCellModelLearning.create_bbo_callback_with_early_termination(
    config, maxiters)

    # create optimization problem
    prob = Optimization.OptimizationProblem(
        obj_func,
        classical_params_array,
        p;
        lb= learning_problem.p_repr_lb,
        ub=learning_problem.p_repr_ub)

    # solve optimization problem
    sol = solve(prob, BBO_adaptive_de_rand_1_bin(); callback=callback, maxiters=maxiters)
    final_params_repr = CombiCellModelLearning.reconstruct_learning_params_from_array(sol.minimizer, p_repr_ig, learning_problem.model)
    final_params_derepr = CombiCellModelLearning.derepresent_all(final_params_repr, learning_problem.model)

    return final_params_derepr, loss_history
end

final_params_derepr, loss_history = bbo_learn(learning_problem, p_repr_ig)

savedir = "/home/xialu/Documents/W25/AllardRotation/CombiCellLocal/experiments/02032026_bicycleHardAccessory/"

# TODO: write function plot_loss_history (1 plot), plot_fit_vs_data(8 plots), plot_error (8 plots), compute_metrics (rmse for each output = 8 rmses)
#TODO : move all the below for hpc later
# Helper function to generate fit data once
function generate_fit_data(fakeData, fitted_params, model)
    """
    Generates model predictions for all outputs using fitted parameters.
    Returns a dictionary with the same structure as fakeData but with fitted values.
    
    Args:
        fakeData: Dictionary containing input data (x, KD)
        fitted_params: Fitted parameters from the model
        
    Returns:
        fitData: Dictionary containing fitted values for all outputs
    """
    # Extract input data
    x_data = fakeData["x"]
    kD_data = fakeData["KD"]
    # p_class = fitted_params.p_classical

    predictions = CombiCellModelLearning.forward_combi(x_data, kD_data, p_class, model)
    
    O1_00_fit = predictions[:, 1]
    O2_00_fit = predictions[:, 2]
    O1_10_fit = predictions[:, 3]
    O2_10_fit = predictions[:, 4]
    O1_01_fit = predictions[:, 5]
    O2_01_fit = predictions[:, 6]
    O1_11_fit = predictions[:, 7]
    O2_11_fit = predictions[:, 8]
    
    # Get predictions for all 8 outputs
    
    
    # Create fitData dictionary with same structure as fakeData
    fitData = Dict{String, Vector{Float64}}(
        "x" => x_data,  # Include inputs for convenience
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

# 1. Plot loss history (unchanged, doesn't need fitData)
function plot_loss_history(loss_history, savedir)
    """
    Plots the loss history from the optimization process.
    
    Args:
        loss_history: Array of loss values at each iteration
        savedir: Directory to save the plot
    """
    p = plot(loss_history, 
             xlabel="Iteration", 
             ylabel="Loss", 
             title="Loss History During Optimization",
             label="Loss",
             linewidth=2,
             yscale=:log10,
             legend=:topright,
             grid=true,
             color=:blue)
    
    save_path = joinpath(savedir, "loss_history.png")
    savefig(p, save_path)
    println("Loss history plot saved to: $save_path")
    
    return p
end

# 2. Plot fit vs data using pre-computed fitData
function plot_fit_vs_data(fakeData, fitData, savedir)
    """
    Creates 8 plots comparing fitted model predictions to simulated data.
    Uses pre-computed fitData to avoid redundant calculations.
    
    Args:
        fakeData: Dictionary containing simulated data
        fitData: Dictionary containing fitted values
        savedir: Directory to save the plots
    """
    # Get x data from either dictionary
    x_data = fakeData["x"]
    
    # Output names for all 8 outputs
    output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    
    # Create a 4x2 grid of plots
    plots = []
    
    for output_name in output_names
        # Extract data and fit
        data = fakeData[output_name]
        fit = fitData[output_name]
        
        # Create individual plot
        p = scatter(x_data, data, 
                    label="Data",
                    markersize=5,
                    marker=:circle,
                    alpha=0.7,
                    xlabel="x",
                    ylabel=output_name,
                    title="Fit vs Data: $output_name",
                    legend=:topleft)
        
        # Add fitted curve
        plot!(p, x_data, fit,
              label="Fit",
              linewidth=2,
              color=:red)
        
        push!(plots, p)
    end
    
    # Combine all plots
    combined_plot = plot(plots..., 
                         layout=(4, 2),
                         size=(1200, 1600),
                         plot_title="Model Fit vs Simulated Data (All Outputs)")
    
    save_path = joinpath(savedir, "fit_vs_data_all_outputs.png")
    savefig(combined_plot, save_path)
    println("Fit vs data plots saved to: $save_path")
    
    return combined_plot
end

# 3. Plot error using pre-computed fitData
function plot_error_all_outputs(fakeData, fitData, savedir)
    """
    Creates 8 error plots showing absolute differences between data and fit.
    Uses pre-computed fitData to avoid redundant calculations.
    
    Args:
        fakeData: Dictionary containing simulated data
        fitData: Dictionary containing fitted values
        savedir: Directory to save the plots
    """
    x_data = fakeData["x"]
    output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    
    plots = []
    
    for output_name in output_names
        # Extract data and fit
        data = fakeData[output_name]
        fit = fitData[output_name]
        
        # Calculate absolute error
        error = abs.(data .- fit)
        
        # Create individual error plot
        p = scatter(x_data, error,
                    label="Absolute Error",
                    markersize=5,
                    marker=:diamond,
                    color=:red,
                    xlabel="x",
                    ylabel="|Data - Fit|",
                    title="Error: $output_name",
                    legend=:topright)
        
        # Add mean error line
        mean_error = mean(error)
        hline!(p, [mean_error],
               label="Mean Error = $(round(mean_error, digits=5))",
               linewidth=2,
               linestyle=:dash,
               color=:blue)
        
        push!(plots, p)
    end
    
    # Combine all plots
    combined_plot = plot(plots..., 
                         layout=(4, 2),
                         size=(1200, 1600),
                         plot_title="Error Plots (All Outputs)")
    
    save_path = joinpath(savedir, "error_plots_all_outputs.png")
    savefig(combined_plot, save_path)
    println("Error plots saved to: $save_path")
    
    return combined_plot
end

# 4. Compute metrics using pre-computed fitData
function compute_metrics_all_outputs(fakeData, fitData, savedir)
    """
    Computes RMSE for each of the 8 output variables.
    Uses pre-computed fitData to avoid redundant calculations.
    
    Args:
        fakeData: Dictionary containing simulated data
        fitData: Dictionary containing fitted values
        savedir: Directory to save the metrics
        
    Returns:
        metrics_dict: Dictionary containing RMSE for each output
    """
    output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    metrics_dict = Dict{String, Float64}()
    
    println("RMSE Metrics:")
    println("-"^40)
    
    # bias = 0
    n = length(fakeData["x"])
    for output_name in output_names
        # Extract data and fit
        data = fakeData[output_name]
        fit = fitData[output_name]
        
        # Calculate RMSE
        rmse = sqrt(mean((data .- fit).^2))
        metrics_dict["RMSE_$output_name"] = rmse

        # Calculate bias 

        bias = abs(sum(fit .> data)/n - 0.5)
        metrics_dict["bias_$output_name"] = bias
        
        # Print formatted output
        println("RMSE_$output_name: $(lpad(round(rmse, digits=6), 12))")
        println("bias_$output_name: $(lpad(round(bias, digits=6), 12))")
    end
   
    
    # Calculate average RMSE
    # avg_rmse = mean(values(metrics_dict))
    # metrics_dict["Average_RMSE"] = avg_rmse
    
    # println("-"^40)
    # println("Average RMSE: $(lpad(round(avg_rmse, digits=6), 16))")
    
    # Save metrics
    metrics_path = joinpath(savedir, "model_metrics.jld2")
    @save metrics_path metrics_dict
    println("\nMetrics saved to: $metrics_path")
    
    return metrics_dict
end

# 5. Additional utility function: Plot residuals (alternative to absolute error)
function plot_residuals(fakeData, fitData, savedir)
    """
    Creates residual plots (data - fit) instead of absolute error.
    Residuals can show systematic patterns in errors.
    
    Args:
        fakeData: Dictionary containing simulated data
        fitData: Dictionary containing fitted values
        savedir: Directory to save the plots
    """
    x_data = fakeData["x"]
    output_names = ["O1_00", "O2_00", "O1_10", "O2_10", "O1_01", "O2_01", "O1_11", "O2_11"]
    
    plots = []
    
    for output_name in output_names
        # Extract data and fit
        data = fakeData[output_name]
        fit = fitData[output_name]
        
        # Calculate residuals (data - fit)
        residuals = data .- fit
        
        # Create residual plot
        p = scatter(x_data, residuals,
                    label="Residuals",
                    markersize=5,
                    marker=:circle,
                    color=:purple,
                    xlabel="x",
                    ylabel="Data - Fit",
                    title="Residuals: $output_name",
                    legend=:topright)
        
        # Add horizontal line at y=0 for reference
        hline!(p, [0.0],
               label="Zero Line",
               linewidth=1,
               linestyle=:dash,
               color=:black,
               alpha=0.5)
        
        # Add mean residual line
        mean_residual = mean(residuals)
        hline!(p, [mean_residual],
               label="Mean Residual = $(round(mean_residual, digits=5))",
               linewidth=2,
               linestyle=:solid,
               color=:red)
        
        push!(plots, p)
    end
    
    # Combine all plots
    combined_plot = plot(plots..., 
                         layout=(4, 2),
                         size=(1200, 1600),
                         plot_title="Residual Plots (All Outputs)")
    
    save_path = joinpath(savedir, "residual_plots_all_outputs.png")
    savefig(combined_plot, save_path)
    println("Residual plots saved to: $save_path")
    
    return combined_plot
end

# Main function to run all plots and metrics
function generate_all_plots_and_metrics(fakeData, fitted_params, loss_history, savedir, model)
    """
    Convenience function to generate all plots and compute metrics.
    Efficiently computes fit data once and reuses it.
    
    Args:
        fakeData: Dictionary containing simulated data
        fitted_params: Parameters from the fitted model
        loss_history: Array of loss values
        savedir: Directory to save all outputs
    """
    println("\n" * "="^50)
    println("Generating Plots and Metrics")
    println("="^50)
    
    # Step 1: Generate fit data ONCE
    println("\n1. Generating model predictions...")
    fitData = generate_fit_data(fakeData, fitted_params, model)
    
    # Step 2: Plot loss history
    println("\n2. Plotting loss history...")
    plot_loss_history(loss_history, savedir)
    
    # Step 3: Plot fit vs data
    println("\n3. Plotting fit vs data...")
    plot_fit_vs_data(fakeData, fitData, savedir)
    
    # # Step 4: Plot error
    # println("\n4. Plotting error...")
    # plot_error_all_outputs(fakeData, fitData, savedir)
    
    # Step 5: Plot residuals
    println("\n5. Plotting residuals...")
    plot_residuals(fakeData, fitData, savedir)
    
    # Step 6: Compute metrics
    println("\n6. Computing metrics...")
    metrics = compute_metrics_all_outputs(fakeData, fitData, savedir)
    
    println("\n" * "="^50)
    println("All plots and metrics generated successfully!")
    println("="^50)
    
    return metrics, fitData
end

# Now use the functions after your optimization:
println("\n=== Starting Plot Generation ===")

# Generate fit data once
p_class = final_params_derepr.p_classical
# fitData = generate_fit_data(fakeData, p_class, model)

# Generate all plots and metrics
all_metrics, fitData = generate_all_plots_and_metrics(
    fakeData, p_class, loss_history, savedir, model
)

# You can also call individual functions:
# plot_fit_vs_data(fakeData, fitData, savedir)
# plot_error_all_outputs(fakeData, fitData, savedir)
# plot_residuals(fakeData, fitData, savedir)
# compute_metrics_all_outputs(fakeData, fitData, savedir)