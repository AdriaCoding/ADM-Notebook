using DifferentialEquations, Flux, DiffEqFlux, Plots

function lotka_volterra(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α*x - β*x*y
    du[2] = dy = -γ*y + δ*x*y
end

# Parameters and initial conditions
params = [1.5, 1.0, 1.0, 1.0]
u0 = Float32[1.0, 1.0]  # Initial conditions for x, y
tspan = (0f0, 5f0)
tstep = 0.5

# Solve the ODE problem
prob = ODEProblem(lotka_volterra, u0, tspan, params)
solution = solve(prob, Tsit5(), saveat=tstep)

# Define a neural network
nn = Flux.Chain(
    Flux.Dense(2, 18, tanh),
    Flux.Dense(18, 2)
)

# Create the Neural ODE problem
neural_p, re = Flux.destructure(nn) # use this p as the initial condition
dudt(u, p, t) = re(p)(u) # need to restructure for backprop
neural_prob = ODEProblem(dudt, u0, tspan, neural_p)

# Define a loss function against the original ODE solution
function loss(p_nn)
    pred = solve(neural_prob, Tsit5(), p=p_nn, saveat=tstep)
    sum(abs2, solution .- pred)
end

# Train the model
opt = ADAM(0.01)
@show loss(neural_p)
Flux.train!(loss, neural_p, Iterators.repeated((), 2000), opt)
@show loss(neural_p)

# Increase timespan to see how well the nn predicts into the future
neural_prob.tspan = (0f0, 20f0)
# Solve with trained neural network parameters
trained_solution = solve(neural_prob, Tsit5(), p=neural_p, saveat=0.1)

# Plot results
scatter(solution, label=["Prey" "Predator"])
plot!(trained_solution, label=["NN Prey" "NN Predator"])