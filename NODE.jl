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

function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, true_values .- pred)
    return loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
callback = function (p, l, pred; doplot = false)
    println(l)
    # plot current prediction against data
    if doplot
        plt = plot(train_years, transpose(pred),
            labels=["x(t)" "y(t)"], ylim=(0, 150))
        scatter!(plt, train_years, true_values[1, :]; 
            label = "hare", color=:green)
        scatter!(plt, train_years, true_values[2, :];
            label = "lynx", color=:red2)

        display(plt)
    end
    return false
end
## Define flux Neural Net
NN = Chain(Dense(2, 16, leakyrelu), Dense(16, 2))
p, st = Lux.setup(rng, NN)
p = p |> ComponentArray

## 'NeuralODE' model
prob_neuralode = NeuralODE(NN, tspan, Tsit5(); saveat = 1)

callback(p, loss_neuralode(p)...; doplot = true)


# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)

result_neuralode = Optimization.solve(optprob, Optimisers.Adam(); callback = callback,
    maxiters = 200)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
    callback, allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot = true)