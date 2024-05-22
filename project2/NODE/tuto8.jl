using ComponentArrays, DiffEqFlux, OrdinaryDiffEq, Optimization, Distributions, Random,
      OptimizationOptimisers, OptimizationOptimJL

nn = Chain(Dense(1, 3, tanh), Dense(3, 1, tanh))
tspan = (0.0f0, 10.0f0)

ffjord_mdl = FFJORD(nn, tspan, (1,), Tsit5(); ad = AutoZygote())
ps, st = Lux.setup(Xoshiro(0), ffjord_mdl)
ps = ComponentArray(ps)
model = StatefulLuxLayer{true}(ffjord_mdl, ps, st)

# Training
data_dist = Normal(6.0f0, 0.7f0)
train_data = Float32.(rand(data_dist, 1, 100))

function loss(θ)
    logpx, λ₁, λ₂ = model(train_data, θ)
    return -mean(logpx)
end

function cb(p, l)
    @info "FFJORD Training" loss=loss(p)
    return false
end

adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ps)

res1 = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.01); maxiters = 20, callback = cb)

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(); allow_f_increases = false, callback = cb)

# Evaluation
using Distances

st_ = (; st..., monte_carlo = false)

actual_pdf = pdf.(data_dist, train_data)
learned_pdf = exp.(ffjord_mdl(train_data, res2.u, st_)[1][1])
train_dis = totalvariation(learned_pdf, actual_pdf) / size(train_data, 2)

# Data Generation
ffjord_dist = FFJORDDistribution(ffjord_mdl, ps, st)
new_data = rand(ffjord_dist, 100)