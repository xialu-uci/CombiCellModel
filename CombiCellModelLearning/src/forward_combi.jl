function forward_combi(x::Vector{Float64}, kD::Vector{Float64}, p_derepresented, model)
    fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max, extraCD2, extraPD1 = p_derepresented
    
    # TODO: modify to do fw of parameter subsets
    p_class = ComponentArray(
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
        O2max=O2max
    ) 

    O1_00, O2_00 = fw(x, kD, p_class, model)
    

    # hardcoded # also a lot of memory to do it this way
    p_class_10 =  ComponentArray(
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
        O2max=O2max
    )
    O1_10, O2_10 = fw(x, kD, p_class_10, model)

    # hardcoded
    p_class_01 = ComponentArray(
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
        O2max=extraPD1
    )
    O1_01, O2_01 = fw(x,kD,p_class_01, model)

    # hardcoded
    p_class_11 = ComponentArray(
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
        O2max=extraPD1
    )
    O1_11, O2_11 = fw(x,kD,p_class_11, model)

    return hcat(O1_00, O2_00, O1_10, O2_10,  O1_01, O2_01, O1_11, O2_11)

end