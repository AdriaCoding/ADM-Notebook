using DifferentialEquations, Flux, DiffEqFlux, Plots
using CSV: read; using DataFrames

## Data handling
rawdata = read("project2/datasets/Leigh1968_harelynx.csv", DataFrame)
df = mapcols(x -> Float64.(x .÷ 1000), rawdata[:,[:hare, :lynx]])
df_train = df[1:45,:]
true_values = transpose(Matrix(df_train))

## Problem conditions
u0 = Float32[df_train.hare[1], df_train.lynx[1]]
tspan = Float32.((0.0, size(df_train)[1]-1))
train_years = rawdata.year[1:45]

# Define a neural network
nn = Flux.Chain(
    Flux.Dense(2, 18, tanh),
    Flux.Dense(18, 2)
)

# Create the Neural ODE problem
neural_p, re = Flux.destructure(nn) # use this p as the initial condition
dudt(u, p, t) = re(p)(u) # need to restructure for backprop
neural_prob = ODEProblem(dudt, u0, tspan, neural_p)
untrained_solution = solve(neural_prob, Tsit5(), p=neural_p, saveat=1.0)


function loss(p_nn)
    pred = Array(solve(neural_prob, Tsit5(), p=p_nn, saveat=1.0))
    sum(abs2, true_values .- pred)
end
loss(neural_p)

callback = function () #callback function to observe training
    pred = Array(solve(neural_prob, Tsit5(), p=neural_p, saveat=1.0))
    display(sum(abs2, true_values .- pred))
    # plot current prediction against data
    pl = scatter(train_years, true_values', label=["Hare" "Lynx"])
    pred
    plot!(train_years, pred', label=["NN Prey" "NN Predator"], ylim=(0, 150))
    display(plot(pl))
end
callback()
# Train the model
opt = Flux.setup(Adam(0.01), nn)
for i in 1:2
    println("Training batch $i")
    Flux.train!(loss, neural_p, Iterators.repeated((), 1000), opt)
    callback()
end
# Solve with trained neural network parameters
trained_solution = solve(neural_prob, Tsit5(), p=neural_p, saveat=0.1)
plot!(trained_solution, label=["NN Prey" "NN Predator"])