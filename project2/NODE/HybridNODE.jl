using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random: Xoshiro; using CSV: read
using Plots, DataFrames
gr()

## Data retirieval
rawdata = read("project2/datasets/Leigh1968_harelynx.csv", DataFrame)
df = mapcols(x -> Float32.(x .÷ 1000), rawdata[:,[:hare, :lynx]])
train_size = 20
df_train = df[1:train_size,:]
train_years = rawdata.year[1:train_size]

# Normalize data
scale = eachcol(df) .|> maximum |> transpose |> Array
normalized_data = Array(df_train./scale)'
normalized_data' .* scale

#Display our data
dataplot = scatter(train_years, normalized_data[1,:], label="Hares", color="blue", lw=2)
scatter!(dataplot, train_years, normalized_data[2,:], label="Lynx", color="red", lw=2)
    ## Problem conditions
rng = Xoshiro(1)
u0 = normalized_data[:,1]
tspan = Float32.((0.0, train_size-1))
t = range(tspan[1], tspan[2], length=train_size) |> Array

## Define the network
# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))

# Define the network 2->5->5->5->2
U = Chain(
    Dense(2,5,rbf), Dense(5,5, tanh), Dense(5,2)
    )
p_nn, state = Lux.setup(rng, U)
const st = state
# ps = [α, β, δ, γ; Network parameters]
p= ComponentArray(NN=p_nn, LV=rand(rng, Float32,4))

# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    û = U(u, p.NN, st)[1] # Forward pass
    α, β, γ, δ = p.LV
    # Lokta-Volterra equations + ANN
    du[1] = α*u[1] - β*u[1]*u[2] + û[1]
    du[2] = γ*u[1]*u[2] - δ*u[2] + û[2]
end
# Define the problem
prob_nn = ODEProblem(ude_dynamics!, u0, tspan, p)

initial_sol = solve(prob_nn, Rosenbrock23(), saveat = t)
function predict(θ, u0=u0, T = t)
    _prob = remake(prob_nn, u0 = u0 , tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Rosenbrock23(), saveat = T,
    abstol = 1e-6, reltol = 1e-6,
    sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

using BenchmarkTools
@btime predd = predict(p)

paint(p, plt = dataplot) = begin
    pred = predict(p)
    predplot = plot(plt, train_years, pred[1,:], label="Hares (pred)",
         color="blue", lw=2, ls=:dash, ylims=(-0.1,1.1))
    plot!(train_years, pred[2,:], label="Lynx (pred)", color="red", lw=2, ls=:dash)
    display(predplot)
end

paint(p)

function loss(p)
    pred = predict(p)
    return sum(abs2, normalized_data .- pred)
end

@time loss(p)
losses = Float32[]

callback = function (p, l; doplot=true)
    push!(losses, l)
    if length(losses) % 100 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
        if doplot
            paint(p.u)
        end
    end
    return false
end

# train this model!!
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)
res1 = Optimization.solve(optprob, ADAM(0.1), callback = callback, maxiters = 1000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback = callback, maxiters = 1000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

result_neuralode = Optimization.solve(optprob, opt,
    maxiters = 40)

optprob2 = remake(optprob; u0 = result_neuralode.u)
callback(result_neuralode.u, loss(result_neuralode.u)...; doplot = true)
result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
    callback, allow_f_increases = false)

callback(result_neuralode2.u, loss(result_neuralode2.u)...; doplot = true)