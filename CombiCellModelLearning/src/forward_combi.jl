function forward_combi(x::Vector{Float64}, kD::Vector{Float64}, p_derepresented, model)
    fI, alpha, tT, g1, k_on_2d, kP, nKP, lambdaX, nC, XO1, O1max, O2max, extraCD2, extraPD1 = p_derepresented
    intPoint1, intPoint2 = model.intPoints

    p_class = ComponentArray(
        fI=fI, alpha=alpha, tT=tT, g1=g1, k_on_2d=k_on_2d,
        kP=kP, nKP=nKP, lambdaX=lambdaX, nC=nC, XO1=XO1,
        O1max=O1max, O2max=O2max
    )

    configs = (
        (nothing,  nothing),   # 00: no replacement
        (extraCD2, nothing),   # 10
        (nothing,  extraPD1),  # 01
        (extraCD2, extraPD1),  # 11
    )

    results = map(configs) do (cd2, pd1)
        p = copy(p_class)
        isnothing(cd2) || (p[intPoint1] = cd2)
        isnothing(pd1) || (p[intPoint2] = pd1)
        fw(x, kD, p, model)
    end

    return hcat((col for (o1, o2) in results for col in (o1, o2))...)
end