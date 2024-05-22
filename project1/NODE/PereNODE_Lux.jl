using DifferentialEquations, Lux, DiffEqFlux, Plots
using Optimization, ComponentArrays, Random
using OptimizationOptimisers: Adam
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
dataplot = scatter(solution, label=["Prey" "Predator"])


# Define a neural network
nn = Chain(
    Dense(2, 18, tanh),
    Dense(18, 2)
)

neural_p, state = Lux.setup(Xoshiro(0), nn)
const state = state # For performance
neural_p = neural_p |> ComponentArray

# Create the (Neural) ODE problem. Note that ∇u=f(u)!=f(u,t)
dudt(u, p, t) = nn(u, p, state)[1]
neural_prob = ODEProblem(dudt, u0, tspan, neural_p)
# 
function predict(θ, u0=u0, tspan = tspan)
    _prob = remake(neural_prob, u0 = u0 , tspan = tspan, p = θ)
    Array(solve(_prob, Tsit5(), saveat = tstep))
end

@time predict(neural_p) 

# Define a loss function against the original ODE solution
function loss(p_nn)
    pred = predict(p_nn)
    sum(abs2, solution .- pred)
end

@time loss(neural_p) 

callback = function (opt_state, l; doplot=true)
    if opt_state.iter % 100 == 0
        println("Iteration $(opt_state.iter) with loss $l")
    end
    if doplot && opt_state.iter % 10 == 0
        p = opt_state.u
        display(plot(dataplot,
            solution.t, predict(p)', ylims=(0, 3),
            label=["NN Prey" "NN Predator"] ))
    end
    return false
end
# Train the model
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)
res1 = Optimization.solve(optprob, Adam(0.01),
                        callback=callback, maxiters = 5000)

trained_p = res1.u

# Test prediction in the future
future_tspan = (0f0, 20f0)
future_real = solve(prob, Tsit5(), saveat=tstep, tspan=future_tspan)
future_nn = remake(neural_prob, tspan = future_tspan, p = trained_p)
future_pred = solve(future_nn, Tsit5(), saveat=tstep/10)
# Plot results
scatter(future_real, label=["Prey" "Predator"])
plot!(future_pred, label=["NNPrey" "NNPredator"], title="Predictions in the future")