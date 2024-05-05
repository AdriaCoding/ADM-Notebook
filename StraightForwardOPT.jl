using DifferentialEquations, Optimization, OptimizationPolyalgorithms, SciMLSensitivity
using ForwardDiff, Plots, CSV, DataFrames

rawdata = CSV.read("datasets/Leigh1968_harelynx.csv", DataFrame)
df = mapcols(x -> Float64.(x .÷ 1000), rawdata[:,[:hare, :lynx]])

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end



# Initial condition
u0 = [21.0, 49.0]

# Simulation interval
tspan = (0.0, Float32(size(df)[1]) - 1)

# LV equation parameter. p = [α, β, δ, γ]
p = [2.0, 0.1, 4.0, 0.1]

# Setup the ODE problem, then solve
#=
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
datasol = solve(prob, saveat = 1)
data = Array(datasol)
=#
datasol = Matrix(df)
data = transpose(Matrix(df))
## Now do the optimization process
function loss(newp)
    newprob = remake(prob, p = newp)
    sol = solve(newprob, saveat = 1)
    loss = sum(abs2, sol .- data)
    return loss, sol
end
pl
callback = function (p, l, sol)
    display(l)
    plt = plot(sol, ylim = (0, 150), label = "Current Prediction")
    scatter!(plt, datasol, label = "data")
    display(plt)
    # Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
    return false
end

adtype = Optimization.AutoZygote()
pguess = [2.1, 0.1, 4.0, 0.1]
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pguess)

 
result_ode = Optimization.solve(optprob, PolyOpt(),
                                callback = callback,
                                maxiters = 20000)

final_ode = ODEProblem(lotka_volterra!, u0, tspan, result_ode.u)
final_sol = solve(final_ode, saveat = 0.01)
plt = plot(final_sol, ylim = (0, 150), label = "Current Prediction")
scatter!(plt, datasol, label = "data")
display(plt)