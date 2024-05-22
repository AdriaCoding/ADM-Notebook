using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random: default_rng; using CSV: read
using Plots, DataFrames

## Data retirieval
rawdata = read("project2/datasets/Leigh1968_harelynx.csv", DataFrame)
df = mapcols(x -> Float32.(x .÷ 1000), rawdata[:,[:hare, :lynx]])
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

## Define the network
# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))

# Define the network 2->5->5->5->2
U = Chain(
    Dense(2,5,rbf), Dense(5,5, rbf), Dense(5,5, tanh), Dense(5,2)
)
p, st = Lux.setup(rng, U)
const state = st
# ps = [α, β, δ, γ; Network parameters]
ps = [rand(Float32,4); p]

# Define the hybrid model
function ude_dynamics!(du,u, p, t)
    û = U(u, p[3:end]) # Network prediction
    α, β, δ, γ = p[1:4]
    # We assume a linear birth rate for the prey
    du[1] = p[1]*u[1] + û[1]
    # We assume a linear decay rate for the predator
    du[2] = -p[2]*u[2] + û[2]
end
U(u0, ps, st)
# Define the problem
prob_nn = ODEProblem(ude_dynamics!, u0, tspan, p)

solve(prob_nn, Vern7())
function predict(p)
    Array(model(u0, p, st)[1])
end

function loss(p)
    pred = predict(p)
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
# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

opt = OptimizationOptimisers.Adam(0.005)
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)
result_neuralode = Optimization.solve(optprob, opt; callback = callback,
    maxiters = 40)

optprob2 = remake(optprob; u0 = result_neuralode.u)
callback(result_neuralode.u, loss(result_neuralode.u)...; doplot = true)
result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
    callback, allow_f_increases = false)

callback(result_neuralode2.u, loss(result_neuralode2.u)...; doplot = true)