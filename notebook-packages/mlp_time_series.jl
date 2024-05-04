# mlp_time_series.jl
using Pkg
Pkg.add("Flux")
Pkg.add("Plots")
using Flux, Plots

# Data Preparation
data = [2, 4, 6, 8, 6, 4, 3, 3, 5, 7]

function create_dataset(data, lookback)
    X, Y = [], []
    for i in lookback+1:length(data)
        push!(X, data[i-lookback:i-1])
        push!(Y, data[i])
    end
    return hcat(X...)', vcat(Y...)
end

lookback = 3
X, Y = create_dataset(data, lookback)

# Model Definition
model = Chain(
    Dense(lookback, 5, relu),
    Dense(5, 1)
)

# Training Setup
loss(x, y) = Flux.mse(model(x), y)
optimizer = ADAM(0.01)
data = [(X, Y)]

# Model Training
epochs = 2000
for epoch in 1:epochs
    Flux.train!(loss, model, data, optimizer)
    if epoch % 100 == 0
        println("Epoch $epoch, Loss: $(loss(X, Y))")
    end
end

# Prediction
last_sequence = data[end-lookback+1:end]
last_sequence = last_sequence'  # Transpose to make it a row vector
next_value_prediction = model(last_sequence)

println("Predicted next value: ", next_value_prediction)

