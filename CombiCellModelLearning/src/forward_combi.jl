function forward_combi(x::Vector{Float64}, kD::Vector{Float64}, p_derepresented, model)
    # fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max = p_derepresented.p_classical 
    
    # TODO: modify to do fw of parameter subsets
    p_class = p_derepresented.p_classical
    O1_00, O2_00 = fw(x, kD, p_class, model)

    # hardcoded # also a lot of memory to do it this way
    p_class_10 = deepcopy(p_class)
    p_class_10[end-1] = p_derepresented.p_extra[1]
    O1_10, O2_10 = fw(x, kD, p_class_10, model)

    # hardcoded
    p_class_01 = deepcopy(p_class)
    p_class_01[end] = p_derepresented.p_extra[2]
    O1_01, O2_01 = fw(x,kD,p_class_10, model)

    # hardcoded
    p_class_11 = deepcopy(p_class_10)
     p_class_11[end] = p_derepresented.p_extra[2]
    O1_11, O2_11 = fw(x,kD,p_class_11, model)

    return hcat(O1_00, O2_00, O1_10, O2_10,  O1_01, O2_01, O1_11, O2_11)

   
   
    # TODO: (later) for loop to do extraCD2 and extraPD1 replacing every other parameter
    # numClass = length(p_class)
    # for i = 1:numClass
    #     p_class[i] = p_derepresented.p_extra[1]
    #     for j = 1:numClass
    #         p_class[j] = p_derepresented.p_extra[2]
    
    # end
    
    #return fw(x, kD, p_derepresented, model)

end