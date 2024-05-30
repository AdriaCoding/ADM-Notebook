using OrdinaryDiffEq, Lux, LuxCUDA, SciMLSensitivity, ComponentArrays, Random
rng = Xoshiro(0)

const cdev = cpu_device()
const gdev = gpu_device()

model = Chain(Dense(2, 50, tanh), Dense(50, 2))
ps, st = Lux.setup(rng, model)
ps = ps |> ComponentArray |> gdev
st = st |> gdev
dudt(u, p, t) = model(u, p, st)[1]

# Simulation interval and intermediary points
tspan = (0.0f0, 10.0f0)
tsteps = 0.0f0:1.0f-1:10.0f0

u0 = Float32[2.0; 0.0] |> gdev
prob_gpu = ODEProblem(dudt, u0, tspan, ps)

# Runs on a GPU
sol_gpu = solve(prob_gpu, Tsit5(); saveat = tsteps)