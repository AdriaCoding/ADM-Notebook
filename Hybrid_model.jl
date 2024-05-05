using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random: default_rng; using CSV: read
using Plots, DataFrames

## Data handling
rawdata = read("datasets/Leigh1968_harelynx.csv", DataFrame)
df = mapcols(x -> Float64.(x .÷ 1000), rawdata[:,[:hare, :lynx]])
df_train = df[1:45,:]
true_values = transpose(Matrix(df_train))

## Problem conditions
rng = default_rng()
u0 = Float32[df_train.hare[1], df_train.lynx[1]]
tspan = Float32.((0.0, size(df_train)[1]-1))
train_years = rawdata.year[1:45]

## Define flux Neural Net
NN = Chain(Dense(2, 5, tanh_fast), Dense(5, 2))
p_NN, st = Lux.setup(rng, NN)
p = [1.0f0, 1.0f0, p_NN]
const _st = st

function hybrid_model(du, u, p, t)
    x, y = u
    α, δ, p_NN = p
    𝕩 , 𝕪 = NN(u, p_NN, _st)[1] #Forward propagate
    du[1] = dx = α * x + 𝕩
    du[2] = dy = -δ * y + 𝕪
end
prob = ODEProblem(hybrid_model, u0, tspan, p)

function predict(θ, X = u0, T = trange)
    _prob = remake(prob, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol = 1e-6, reltol = 1e-6,
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end
predict(p)

function loss_neuralode(p)
    pred = predict(p)
    loss = sum(abs2, true_values .- pred)
    return loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
callback = function (p, l, pred; doplot = true)
    println(l)
    # plot current prediction against data
    if doplot
        plt = plot(train_years, transpose(pred),
            labels=["x(t)" "y(t)"],
            ylims = (0,150))
        scatter!(plt, train_years, true_values[1, :]; 
            label = "hare", color=:green)
        scatter!(plt, train_years, true_values[2, :]; 
            label = "lynx", color=:red2)

        display(plt)
    end
    return false
end

pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot = true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float32}(p))

using Functors
nt = (;a  = randn(10), b = randn(2))
fmap(copy, nt)


result_neuralode = Optimization.solve(optprob, Optimisers.Adam(0.05); callback = callback,
    maxiters = 200)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
    callback, allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot = true)