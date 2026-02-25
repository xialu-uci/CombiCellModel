function forward_combi(x::Vector{Float64}, kD::Vector{Float64}, p_derepresented, model)
    intPoint1, intPoint2 = model.intPoints

    if isnothing(intPoint1) && isnothing(intPoint2)
        return forward_simple(x, kD, p_derepresented, model)
    end

    fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max, extraCD2, extraPD1 = p_derepresented
    p_class = ComponentArray(
        fI=fI, alpha=alpha, tT=tT, g1=g1, k_on_2d=k_on_2d,
        kP=kP, nKP=nKP, lambdaX=lambdaX, nC=nC, XO1=XO1,
        O1max=O1max, O2max=O2max
    )
    configs = (
        (nothing,  nothing),
        (extraCD2, nothing),
        (nothing,  extraPD1),
        (extraCD2, extraPD1),
    )
    results = map(configs) do (cd2, pd1)
        p = copy(p_class)
        isnothing(cd2) || (p[intPoint1] = cd2)
        isnothing(pd1) || (p[intPoint2] = pd1) # curr handling for equal: pd1 overrides cd2
        fw(x, kD, p, model)
    end
    return hcat((col for (o1, o2) in results for col in (o1, o2))...)
end

function forward_simple(x::Vector{Float64}, kD::Vector{Float64}, p_derepresented, model)
    fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max = p_derepresented
    p_class = ComponentArray(
        fI=fI, alpha=alpha, tT=tT, g1=g1, k_on_2d=k_on_2d,
        kP=kP, nKP=nKP, lambdaX=lambdaX, nC=nC, XO1=XO1,
        O1max=O1max, O2max=O2max
    )
    O1, O2 = fw(x, kD, p_class, model)
    return hcat(O1, O2)
end