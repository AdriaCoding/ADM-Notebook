using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random: default_rng; using CSV: read
using Plots, DataFrames

## Data retirieval
rawdata = read("project1/datasets/Leigh1968_harelynx.csv", DataFrame)
df = mapcols(x -> Float64.(x .÷ 1000), rawdata[:,[:hare, :lynx]])
train_size = 45
df_train = df[1:train_size,:]
train_years = rawdata.year[1:train_size]

# Normalize data
scale = eachcol(df) .|> maximum |> transpose |> Array
normalized_data = Array(df_train./scale)'
normalized_data' .* scale

## Problem conditions
rng = default_rng()
u0 = normalized_data[:,1]
tspan = Float32.((0.0, size(df_train)[1]-1))

function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, normalized_data .- pred)
    return loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
callback = function (p, l, pred; doplot = true)
    println(l)
    # plot current prediction against data
    if doplot
        plt = plot(train_years, transpose(pred),
            labels=["x(t)" "y(t)"], ylim=(-0.1, 1.1))
        scatter!(plt, train_years, normalized_data[1, :]; 
            label = "hare", color=:green)
        scatter!(plt, train_years, normalized_data[2, :];
            label = "lynx", color=:red2)

        display(plt)
    end
    return false
end
## Define flux Neural Net
NN = Chain(x->[x;x[1]*x[2]],Dense(3, 5, tanh), Dense(5, 5, relu), Dense(5,2))
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
    maxiters = 400)

optprob2 = remake(optprob; u0 = result_neuralode.u)
callback(result_neuralode.u, loss_neuralode(result_neuralode.u)...; doplot = true)
result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
    callback, allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot = true)