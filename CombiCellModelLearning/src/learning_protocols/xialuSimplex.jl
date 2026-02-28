
function simplex_learn(learning_problem, p_repr, intPoints)

    function obj_func(x, p)
        p_repr = CombiCellModelLearning.reconstruct_learning_params_from_array(x, p_repr_ig,learning_problem.model) # this is where params are updated # the trick is the x is the actual params we want. 
        # only pass through p_repr_ig for the keys
        return CombiCellModelLearning.get_loss(p_repr, intPoints; learning_problem=learning_problem)
    end
    # initial guess params array
    classical_params_array = collect(values(copy(p_repr_ig)))

    # p
    p = [1.0, 100.0] # not used in obj_func, honestly I think i could delete this but will leave in for now and see if it changes anything later

    # algo is the optimization algorithm (here bbo)
    # maxiters is maximum iterations
    maxiters = 300000 # not reduced for testing?
    # callback is a function called at each iteration, s.t. optimzation stops if it returns true
    config = CallbackConfig() # just stores info for callback function in fields

    loss_history = Float32[]
    parameter_history = logging ? [] : nothing # generally not loging
    
    function callback(p, lossval)
        push!(loss_history, lossval)
        current_iter = length(loss_history)
        
        if callback_config.verbose && current_iter % protocol.print_frequency == 0
            qdrms = sqrt(lossval / callback_config.constants.qdrms_divisor)
            println("In simplex, current loss after $current_iter iterations: $lossval, qdrms=$qdrms at $(now())")
        end
        
        if logging
            push!(parameter_history, deepcopy(p))
        end
        
        return false
    end
    
    prob = Optimization.OptimizationProblem(
        obj_func,
        classical_params_array,
        p;
        lb= learning_problem.p_repr_lb,
        ub=learning_problem.p_repr_ub)

    opt = NLopt.LN_SBPLX()

    # hardcoding solve options for now, can make more flexible later if needed
    solve_options = Dict{Symbol, Any}()
    solve_options[:reltol] = 1e-6 
    solve_options[:abstol] = 1e-6

    sol = solve(prob, opt; callback=callback, maxiters=maxiters, solve_options...)

    final_params_repr = CombiCellModelLearning.reconstruct_learning_params_from_array(sol.minimizer, p_repr_ig, learning_problem.model)
    final_params_derepr = CombiCellModelLearning.derepresent_all(final_params_repr, intPoints, learning_problem.model)

    return final_params_derepr, loss_history


end