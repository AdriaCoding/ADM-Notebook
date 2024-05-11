using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random: default_rng; using CSV: read
using Plots, DataFrames

## Data handling
rawdata = read("datasets/Leigh1968_harelynx.csv", DataFrame)
df = mapcols(x -> Float64.(x .÷ 1000), rawdata[:,[:hare, :lynx]])
df_train = df[1:45,:]
u_train = transpose(Matrix(df_train))
t_train = Array(0.0:Float64(size(df_train)[1]-1))
train_years = rawdata.year[1:45]

## Problem conditions
rng = default_rng()
u0 = Float64[df_train.hare[1], df_train.lynx[1]]

function neural_ode(t, data_dim)
    f = Lux.Chain(
        Lux.Dense(data_dim, 32, swish),
        Lux.Dense(32, 64, swish),
        Lux.Dense(64, data_dim)
    )

    node = NeuralODE(
        f, extrema(t), Tsit5(),
        saveat=t,
        abstol=1e-9, reltol=1e-9
    )
    
    rng = default_rng()
    p, state = Lux.setup(rng, f)

    return node, ComponentArray(p), state
end

function train_one_round(node, p, st, y, maxiters, lr)
    pred(p) = Array(node(u0, p, st)[1])
    loss(p) = sum(abs2, pred(p) .- y)/length(u_train)
    callback(p, l) = begin
        println(l)
        plot_trajectories(y, pred(p))
        return false
    end
    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((p, _ ) -> loss(p), adtype)
    optprob = OptimizationProblem(optf, p)
    callback(p, loss(p))
    res = solve(optprob, Optimisers.ADAMW(lr), maxiters=maxiters)
    return res.u, st, pred(p)
end
    
function train(y, t, maxiters = 150, lr = 1e-2)
    p=nothing
    state=nothing
    
    for k in 3:3:length(t_train)
        println("Training batch of first $k values")
        node, p_new, state_new = neural_ode(t[1:k], size(y, 1))
        if p === nothing p = p_new end
        if state === nothing state = state_new end

        p, state, pred = train_one_round( node, p, state, y[:,1:k], maxiters, lr)
        plot_trajectories(y, pred)
    end
    p, state
end


predict(y0, t, p=nothing, state=nothing) = begin
    node, p_random, state_random = neural_ode(t, length(y0))
    if p === nothing p = p_random end
    if state === nothing state = state_random end
    ŷ = Array(node(y0, p, state)[1])
end
function plot_trajectories(y, pred)
    n = size(y, 2)
    m = size(pred,2)
    years = rawdata.year
    plt = plot(years[1:m], pred',
    labels=["x(t)" "y(t)"], ylim=(0, 150))
    scatter!(plt, years[1:n], y[1, :]; 
    label = "hare", color=:green)
    scatter!(plt, years[1:n], y[2, :];
    label = "lynx", color=:red2)
    
    display(plt)
end

prediction = predict(u0, t_train)
plot_trajectories(u_train, prediction)

res = train(u_train[:,1:27], t_train[1:27])