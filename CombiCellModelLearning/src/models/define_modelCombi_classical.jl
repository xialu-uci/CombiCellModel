struct ModelCombiClassic <: AbstractClassicalModel
    p_classical_derepresented_ig::ComponentArray{Float64} # classical parameters
    p_derepresented_lowerbounds::ComponentArray{Float64} # lower bounds for derepresented parameters
    p_derepresented_upperbounds::ComponentArray{Float64} # upper bounds for derepresented parameters
    u0::Vector{Float64}  # Not used for algebraic model, but kept for compatibility for now...
    params_repr_ig::ComponentArray{Float64} # biophysical parameters mapped to spaces suitable for optimization # log, logit, sqrt transforms
    params_derepresented_ig::ComponentArray{Float64}
end

function make_ModelCombiClassic(;)
   # close initial guess 
p_classical_derepresented_ig = ComponentArray(
        fI=0.5,
        alpha=1.9e6,
        tT=600,
        g1=0.8,
        k_on_2d=14.0, #(12,16)
        kP=0.5, #(0.3,0.7)
        nKP=2.0, #(1.5,2.5)
        lambdaX=0.08, #(0.03,0.1)
        nC=2.0, #(1.5,2.5)
        XO1=0.5, #(0.3,0.7)
        O1max=0.8, #(0.7,1.0)
        O2max=100.0 #(80,120)
    )


    # tight bounds around true params

    p_derepresented_lowerbounds = ComponentArray(
        fI=0.3,
        alpha=1.8e6,
        tT=400,
        g1=0.6,
        k_on_2d=12.0,
        # kD=1.0, # TODO: this is given, figure out how to get it in here
        kP=0.3,
        nKP=1.5,
        lambdaX=0.03,
        nC=1.5 ,
        XO1=0.3,
        O1max=0.7,
        O2max=80.0,
    )

    p_derepresented_upperbounds = ComponentArray(
        fI= 0.7,
        alpha=2.1e6,
        tT=600.0,
        g1=0.9,
        k_on_2d=16.0,
        # kD=1e3, # TODO: this is given, figure out how to get it in here
        kP=0.7,
        nKP=2.5,
        lambdaX=0.1,
        nC=2.5,
        XO1=0.7,
        O1max=1.0,
        O2max=120.0,
    )
    # initial condition - not used
    u0 = [0.0]

    params_repr_ig = ComponentArray(
        p_classical=represent_on_type(p_classical_derepresented_ig, ModelCombiClassic),
        # no flex
        
    )

    params_derepresented_ig = ComponentArray(
        p_classical=deepcopy(p_classical_derepresented_ig),
        # no flex
        
    )

    return ModelCombiClassic(
        p_classical_derepresented_ig,
        p_derepresented_lowerbounds,
        p_derepresented_upperbounds,
        u0,
        params_repr_ig,
        params_derepresented_ig,
    )
end

function fw(x::Vector{Float64}, kD::Vector{Float64}, p_class, model::ModelCombiClassic)

# for no accessory condition p_class is just p_derepresented.p_classical
    fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max = p_class

    O1 = Float64[]
    O2 = Float64[]

    # eval_flex(x, p) = FlexiFunctions.evaluate_decompress(x, view(p, 1:length(p)))

    for (xi, kDi) in zip(x, kD)
        CT = (1-fI)*(alpha * xi + tT + g1 * kDi / k_on_2d - sqrt((alpha * xi + tT + g1 * kDi / k_on_2d)^2 - 4 * alpha * xi * tT)) / 2
        CN = (1 / (1 + g1 * kDi / kP))^nKP * CT
        X = CN^nC / (lambdaX^nC + CN^nC)

        O1_val = XO1 / (XO1 + X)
        O2_val = X

        O1i = O1max * O1_val 
        O2i = O2max * O2_val

        push!(O1, O1i)
        push!(O2, O2i)
    end
    return O1, O2
end

# made forward_combi.jl separate file
# function forward_combi(x::Vector{Float64}, kD::Vector{Float64}, p_derepresented)
#     fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max = p_derepresented.p_classical

#     O1 = Float64[]
#     O2 = Float64[]

#     eval_flex(x, p) = FlexiFunctions.evaluate_decompress(x, view(p, 1:length(p)))

#     for (xi, kDi) in zip(x, kD)
#         CT = (alpha * xi + tT + g1 * kDi / k_on_2d - sqrt((alpha * xi + tT + g1 * kDi / k_on_2d)^2 - 4 * alpha * xi * tT)) / 2
#         CN = (1 / (1 + g1 * kDi / kP))^nKP * CT
#         X = CN^nC / (lambdaX^nC + CN^nC)

#         O1_val = XO1 / (XO1 + X)
#         O2_val = X

#         O1i = O1max * abs(eval_flex(abs(O1_val), p_derepresented.flex1_params)) 
#         O2i = O2max * abs(eval_flex(abs(O2_val), p_derepresented.flex2_params))

#         push!(O1, O1i)
#         push!(O2, O2i)
#     end
#     return O1, O2

# end

function represent_on_type(p_derepresented, model_by_type::Type{ModelCombiClassic})
    # initial transformations, subject to change
    return ComponentArray(
        fI=log(p_derepresented.fI),  # log
        alpha=log(p_derepresented.alpha),
        tT=log(p_derepresented.tT),
        g1=log(p_derepresented.g1),
        k_on_2d=log(p_derepresented.k_on_2d),
       #  kD=log(p_derepresented.kD),
        kP=log(p_derepresented.kP),
        nKP=sqrt(p_derepresented.nKP),
        lambdaX=log(p_derepresented.lambdaX),
        nC=sqrt(p_derepresented.nC),
        XO1=log(p_derepresented.XO1),
        O1max=log(p_derepresented.O1max),  # log
        O2max=log(p_derepresented.O2max),
    )
end

function derepresent(p_repr, model::ModelCombiClassic)
    return ComponentArray(
        fI=1 / (1 + exp(-p_repr.fI)),  # sigmoid
        alpha=exp(p_repr.alpha),
        tT=exp(p_repr.tT),
        g1=exp(p_repr.g1),
        k_on_2d=exp(p_repr.k_on_2d),
       #  kD=exp(p_repr.kD),
        kP=exp(p_repr.kP),
        nKP=(p_repr.nKP)^2,
        lambdaX=exp(p_repr.lambdaX),
        nC=(p_repr.nC)^2,
        XO1=exp(p_repr.XO1),
        O1max=1 / (1 + exp(-p_repr.O1max)),  # sigmoid
        O2max=exp(p_repr.O2max),
    )
end