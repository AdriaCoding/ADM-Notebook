using DiffEqFlux, OrdinaryDiffEq, Statistics, LinearAlgebra, Plots, LuxCUDA, Random
using MLUtils, ComponentArrays
using Optimization, OptimizationOptimisers, IterTools

const cdev = cpu_device()
const gdev = gpu_device()

function random_point_in_sphere(dim, min_radius, max_radius)
    distance = (max_radius - min_radius) .* (rand(Float32, 1) .^ (1.0f0 / dim)) .+
               min_radius
    direction = randn(Float32, dim)
    unit_direction = direction ./ norm(direction)
    return distance .* unit_direction
end

function concentric_sphere(dim, inner_radius_range, outer_radius_range,
        num_samples_inner, num_samples_outer; batch_size = 64)
    data = []
    labels = []
    for _ in 1:num_samples_inner
        push!(data, reshape(random_point_in_sphere(dim, inner_radius_range...), :, 1))
        push!(labels, ones(1, 1))
    end
    for _ in 1:num_samples_outer
        push!(data, reshape(random_point_in_sphere(dim, outer_radius_range...), :, 1))
        push!(labels, -ones(1, 1))
    end
    data = cat(data...; dims = 2)
    labels = cat(labels...; dims = 2)
    return DataLoader((data |> gdev, labels |> gdev); batchsize = batch_size,
        shuffle = true, partial = false)
end

diffeqarray_to_array(x) = gdev(x.u[1])

function construct_model(out_dim, input_dim, hidden_dim, augment_dim)
    input_dim = input_dim + augment_dim
    node = NeuralODE(
        Chain(Dense(input_dim, hidden_dim, relu),
            Dense(hidden_dim, hidden_dim, relu), Dense(hidden_dim, input_dim)),
        (0.0f0, 1.0f0),
        Tsit5();
        save_everystep = false,
        reltol = 1.0f-3,
        abstol = 1.0f-3,
        save_start = false)
    node = augment_dim == 0 ? node : AugmentedNDELayer(node, augment_dim)
    model = Chain(node, diffeqarray_to_array, Dense(input_dim, out_dim))
    ps, st = Lux.setup(Xoshiro(0), model)
    return model, ps |> gdev, st |> gdev
end

function plot_contour(model, ps, st, npoints = 300)
    grid_points = zeros(Float32, 2, npoints^2)
    idx = 1
    x = range(-4.0f0, 4.0f0; length = npoints)
    y = range(-4.0f0, 4.0f0; length = npoints)
    for x1 in x, x2 in y
        grid_points[:, idx] .= [x1, x2]
        idx += 1
    end
    sol = reshape(model(grid_points |> gdev, ps, st)[1], npoints, npoints) |> cdev

    return contour(x, y, sol; fill = true, linewidth = 0.0)
end

loss_node(model, x, y, ps, st) = mean((first(model(x, ps, st)) .- y) .^ 2)

dataloader = concentric_sphere(
    2, (0.0f0, 2.0f0), (3.0f0, 4.0f0), 2000, 2000; batch_size = 256)

iter = 0
cb = function (ps, l)
    global iter
    iter += 1
    if iter % 10 == 0
        @info "Augmented Neural ODE" iter=iter loss=l
    end
    return false
end

model, ps, st = construct_model(1, 2, 64, 0)
opt = OptimizationOptimisers.Adam(0.005)

loss_node(model, dataloader.data[1], dataloader.data[2], ps, st)

println("Training Neural ODE")

optfunc = OptimizationFunction(
    (x, p, data, target) -> loss_node(model, data, target, x, st),
    Optimization.AutoZygote())
optprob = OptimizationProblem(optfunc, ComponentArray(ps |> cdev) |> gdev)
res = solve(optprob, opt, IterTools.ncycle(dataloader, 5); callback = cb)

plt_node = plot_contour(model, res.u, st)

model, ps, st = construct_model(1, 2, 64, 1)
opt = OptimizationOptimisers.Adam(0.005)

println()
println("Training Augmented Neural ODE")

optfunc = OptimizationFunction(
    (x, p, data, target) -> loss_node(model, data, target, x, st),
    Optimization.AutoZygote())
optprob = OptimizationProblem(optfunc, ComponentArray(ps |> cdev) |> gdev)
res = solve(optprob, opt, IterTools.ncycle(dataloader, 5); callback = cb)

plot_contour(model, res.u, st)