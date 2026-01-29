# attempt 1: forward solve combicell system withFlexiFunctions

using FlexiFunctions

# define params

# mod1
fI = 0.5 # inhibited TCR can seq pMHC
#fA = 1 - f_I # activated
alpha = 1e6 # conversion added conc to combicells, surface density
tT = 1e3 # TCR surface density
g1 = 1 # conv dissoc to attachment
k_on_2d = 30 # on rate
kD = 30 # 3d dissoc rate


# mod2
kP = 1.0 # proofreading rate
nKP = 2.7 # num proofreading steps

# mod3
lambdaX = 0.05 # threshold for cytoplasmic switch
nC = 4 # switch coeff

# mod4
XO1 = 1 # sat  if X for o1
O1max = 0.95 # max o1
O2max = 150 # "max" o2 

# CT = (alpha * x + tT + g1 *kD/k_on_2d - (( (alpha * x + tT + g1 *kD/k_on_2d)^2 - 4 * alpha * x * tT )^0.5) ) / 2 # mod 1

# CN = (1/(1+g1*kD/kP))^nKP * CT # mod 2 

# X = CN^nC / (lambdaX^nC + CN^nC) # mod 3

# O1 = X / (XO1 + X) # mod 4
# O2 = X # mod 4

# x = input conc of pMHC in combicell prep, 

x = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0] # example inputs

# compute outputs

O1 = []
O2 = [] 

flex1_params = FlexiFunctions.generate_flexi_ig(5) 
flex2_params = FlexiFunctions.generate_flexi_ig(5) # example flexi params

# helper: create the right SubArray type and avoid repeating evaluate_decompress
eval_flex(x, p) = FlexiFunctions.evaluate_decompress(x, view(p, 1:length(p)))

for xi in x
    CT = (alpha * xi + tT + g1 *kD/k_on_2d - (( (alpha * xi + tT + g1 *kD/k_on_2d)^2 - 4 * alpha * xi * tT )^0.5) ) / 2 # mod 1

    CN = (1/(1+g1*kD/kP))^nKP * CT # mod 2 # change to avoid division

    X = CN^nC / (lambdaX^nC + CN^nC) # mod 3

    # O1i = X / (XO1 + X) # mod 4
    # O2i = X # mod 4 # X
    # O1i = abs(FlexiFunctions.evaluate_decompress(abs(X), @view(flex1_params[:]))) / (XO1 + abs(FlexiFunctions.evaluate_decompress(abs(X), @view(flex1_params[:]))) ) # mod 4 with flexi
    # O2i = O2max *abs(FlexiFunctions.evaluate_decompress(abs(X), @view(flex2_params[:]))) # mod
    O1i = O1max * abs(eval_flex(abs(X), flex1_params)) / (XO1 + abs(eval_flex(abs(X), flex1_params)) ) # mod 4 with flexi
    O2i = O2max *abs(eval_flex(abs(X), flex2_params)) # mod 4 with flexi
    push!(O1, O1i)
    push!(O2, O2i)
end

println("Outputs O1: ", O1)
println("Outputs O2: ", O2)


#
# plot results
using Plots
plot(x, O1, xaxis=:log, yaxis=:linear, xlabel="pMHC conc", ylabel="O1 (frac pos cells)", title="Output O1 vs pMHC conc", label="O1")
plot(x, O2, xaxis=:log, yaxis=:linear, xlabel="pMHC conc", ylabel="O2 (cytokine production)", title="Output O2 vs pMHC conc", label="O2")

# returns rhs, suitable for inclusion in ODEProblem # copied from define_modelF8.jl (pulsatile system with flexi functions)
# function make_rhs(i_func, model::ModelF8)

#     # DI-IFFL with flexi1 in m-PP and flexi2 in m-DI

#     function rhs(du, u, p_all_derepresented, t)
#         a, m, w, c = u
#         ta, tm, tw, tc, nwm, nam = p_all_derepresented.p_classical

#         du[1] = (1 / ta) * (i_func(t) - a)
#         du[2] =
#             (1 / tm) * (
#                 + 1 / (1 + abs(FlexiFunctions.evaluate_decompress(abs(w), p_all_derepresented.flex1_params))) * a
#                 - 1 / (1 + abs(FlexiFunctions.evaluate_decompress(abs(a), p_all_derepresented.flex2_params))) * m
#             )
#         du[3] = (1 / tw) * (a - w)
#         du[4] = (m - c / tc)

#         return nothing
#     end

#     return rhs
# end

# inputs: x (conc of pMHC in combicell Prep)
# outputs (predict based on inputs, known from data): measureable output O1 (frac pos cells), measurable output O2 (cytokine production)

# want to find parameters that map inputs to outputs

# define params

# define eqns



# compute o1 and o2 from eqns

# rand input --> output

