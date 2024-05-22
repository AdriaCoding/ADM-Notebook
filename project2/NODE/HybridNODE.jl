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
const normalized_data = Array(df_train./scale)'
normalized_data' .* scale

#Display our data
dataplot = scatter(train_years, normalized_data[1,:], label="Hares", color="blue", lw=2)
scatter!(dataplot, train_years, normalized_data[2,:], label="Lynx", color="red", lw=2)

## Problem conditions
rng = Xoshiro(0)
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
# p.LV = [α, β, δ, γ]
p = ComponentArray(NN=p_nn, LV=rand(rng, Float32,4))

# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    û = U(u, p.NN, st)[1] # Forward pass
    α, β, γ, δ = p.LV
    # Lokta-Volterra equations + ANN
    du[1] = α*u[1] - β*u[1]*u[2] + û[1]
    du[2] = γ*u[1]*u[2] - δ*u[2] + û[2]
end
prob_nn = ODEProblem(ude_dynamics!, u0, tspan, p)

initial_sol = solve(prob_nn, Rosenbrock23(), saveat = t)

function predict(θ; ODEalg = AutoTsit5(Rodas5()), u0=u0, T = t)
    _prob = remake(prob_nn, u0 = u0 , tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, ODEalg, saveat = T,
    abstol = 1e-6, reltol = 1e-6,
    sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

# Test the speed of several stiff ODE solvers.
algs = [Rosenbrock23(), Rodas4(), Rodas5(), AutoTsit5(Rosenbrock23()),  AutoTsit5(Rodas4()), AutoTsit5(Rodas5())];
# To see reliable reults, run this line more than once.
[@elapsed predict(p; ODEalg= alg) for alg in algs]'

# Re-use the plot of the training data to paint new predicitons on top of it.
# Set up an increased time span to see the predictions more smoothly.
resolution = 8
t8 = range(tspan[1], tspan[2], length=train_size*resolution) |> Array
plot_years = range(rawdata.year[1], rawdata.year[train_size], length=train_size*resolution)
paint(p, plt = dataplot) = begin
    pred = predict(p; T=t8)
    predplot = plot(plt, plot_years, pred[1,:], label="Hares (pred)",
         color="blue", ylims=(-0.1,1.1))
    plot!(plot_years, pred[2,:], label="Lynx (pred)", color="red")
    display(predplot)
end

paint(p)

function loss(p)
    pred = predict(p)
    return sum(abs2, normalized_data .- pred)
end

losses = Float32[]

callback = function (opt_state, l; doplot=true)
    if opt_state.iter % 100 == 0
        println("Current loss after $(opt_state.iter) iterations: $(l)")
    end
    if opt_state.iter % 10 == 0
        push!(losses, l)
        if doplot
            p = opt_state.u
            paint(p)
        end
    end
    return false
end

# train this model!!
maxiters = 500

adtype = Optimization.AutoZygote();
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, p);
res1 = Optimization.solve(optprob, ADAM(0.1), callback = callback, maxiters = maxiters);
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.u);
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback = callback, maxiters = maxiters);
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Set final value for the trained parameters

p_trained = res2.u

#Compare with test data 
test_size = size(df, 1)
begin
    # First, plot the data 
    test_data = df ./ scale |> Array |> transpose
    scatter(rawdata.year, test_data[1,:], label="Hares", lw=2)
    scatter!(rawdata.year, test_data[2,:], label="Lynx", lw=2)

    # Make new predictions 
    test_t = Array(range(0.0f0, test_size-1 |> Float32, length=test_size*resolution))
    test_years = test_t .+ rawdata.year[1] 
    test_pred = predict(p_trained, AutoTsit5(Rodas5()), test_data[:,1], test_t)
    sep = length(plot_years)
    test_pred[1,1:sep]
    plot!(test_years[1:sep], test_pred[1,1:sep], label="Hares (pred)", color="blue", lw=1)
    plot!(test_years[1:sep], test_pred[2,1:sep], label="Lynx (pred)", color="red", lw=1)

    test_pred[1,sep:end]
    test_years[sep:end]
    plot!(test_years[sep:end], test_pred[1,sep:end], label=nothing, color="blue", lw=1, linestyle=:dash)
    plot!(test_years[sep:end], test_pred[2,sep:end], label=nothing, color="red", lw=1, linestyle=:dash)
    title!("Hares and Lynx population")
end