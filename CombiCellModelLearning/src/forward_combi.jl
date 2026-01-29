function forward_combi(x::Vector{Float64}, kD::Vector{Float64}, p_derepresented)
    # fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max = p_derepresented.p_classical

    # O1 = Float64[]
    # O2 = Float64[]

    # eval_flex(x, p) = FlexiFunctions.evaluate_decompress(x, view(p, 1:length(p)))

    # for (xi, kDi) in zip(x, kD)
    #     CT = (alpha * xi + tT + g1 * kDi / k_on_2d - sqrt((alpha * xi + tT + g1 * kDi / k_on_2d)^2 - 4 * alpha * xi * tT)) / 2
    #     CN = (1 / (1 + g1 * kDi / kP))^nKP * CT
    #     X = CN^nC / (lambdaX^nC + CN^nC)

    #     O1_val = XO1 / (XO1 + X)
    #     O2_val = X

    #     O1i = O1max * abs(eval_flex(abs(O1_val), p_derepresented.flex1_params)) 
    #     O2i = O2max * abs(eval_flex(abs(O2_val), p_derepresented.flex2_params))

    #     push!(O1, O1i)
    #     push!(O2, O2i)
    # end
    # return O1, O2
    return fw(x, kD, p_derepresented)

end