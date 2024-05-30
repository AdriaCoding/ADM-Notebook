using Lux, ComponentArrays, DifferentialEquations, Random, Plots
using Zygote
using Optimization, OptimizationOptimisers, OptimizationOptimJL 

## Data generation
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

# Define the experimental parameter
tspan = (0.0,3.0)
u0 = [0.44249296,4.6280594]
p_ = [1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0,tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 0.1)

# Ideal data
X = Array(solution)
t = solution.t
DX = Array(solution(solution.t, Val{1}))

# Add noise in terms of the mean
noise_magnitude = 5e-2
data = X .+ (noise_magnitude) .* randn(eltype(X), size(X))

plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(data), color = :red, label = ["Noisy Data" nothing])

## Hybrid model
# Define the network 2->5->2
NN = Chain(
    Dense(2,5,tanh), Dense(5,2)
)
p_nn, st = Lux.setup(Xoshiro(0), NN)
# p = [DiffEq params; Network parameters]
p = [rand(Float32,4); p_nn] |> ComponentArray

# Define the hybrid model
function ude_dynamics!(du,u, p, t)
    û = NN(u, p[5], st)[1] # Forward pass
    α, β, γ, δ = p[1:4]
    # Lokta-Volterra equations + ANN
    du[1] = α*u[1] - β*u[1]*u[2] + û[1]
    du[2] = γ*u[1]*u[2] - δ*u[2] + û[2]
end
prob_nn = ODEProblem(ude_dynamics!,data[:, 1], tspan, p)
function predict(θ, u0 = data[:,1], T = t)
    _prob = remake(prob_nn, u0 = u0, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                ))
end

plot!(solve(prob_nn, Tsit5()), alpha = 0.75, color = :blue, label = ["First solution" nothing])
scatter!(t, predict(p)', color = :blue, label = ["First solution" nothing])


## Training

function loss(θ)
    X̂ = predict(θ)
    sum(abs2, data .- X̂)
end
loss(p)

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)
typeof(p[5])
import Base: real
real(p::NamedTuple{T}) where T = p 
res1 = Optimization.solve(optprob, ADAM(0.1), maxiters = 10)
println("Training loss after $(length(losses)) iterations: $(losses[end])")