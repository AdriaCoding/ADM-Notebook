### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 533f45ff-47c9-479d-8e4c-c81c33cc4e0f
begin
	using Flux, DataFrames, GLMakie, Images, FileIO, ImageShow, DifferentialEquations, ModelingToolkit, PlutoUI, Optimization, ComponentArrays, OptimizationPolyalgorithms,  OptimizationOptimJL, OptimizationOptimisers, ForwardDiff, Lux, Random, DiffEqFlux
	using Statistics: mean; using CSV: read
end

# ╔═╡ d0e5a39f-2da0-410f-a9be-009b039ec891
PlutoUI.TableOfContents()

# ╔═╡ babeac0c-aac9-4725-88b7-6a8d9008d6c0
md"# MLP - Basic model"

# ╔═╡ a3b24f11-5102-47a4-bad2-ac6b50d78a7b
md"Needed libraries: Flux -> ML Julia library -- for the NNs"

# ╔═╡ a9b0ae89-13d4-48b3-8e98-e9a1a714b0b4
md"## Dataset preparation"

# ╔═╡ 5aa263f7-fb27-43f2-abe9-733140d8a0ca
md"#### Load the dataset 
[Hare & Lynx dataset](https://gist.github.com/michaelosthege/27315631c1aedbe55f5affbccabef1ca), a time series dataset of the population of two species; Hudson Bay company Lynx-Hare dataset from Leigh 1968, parsed from paper copy [http://katalog.ub.uni-heidelberg.de/titel/66489211](http://katalog.ub.uni-heidelberg.de/titel/66489211)"

# ╔═╡ e920cd2f-c996-4e67-81ab-5e98fb17d386
rawdata = read("../datasets/Leigh1968_harelynx.csv", DataFrame)

# ╔═╡ 336d223c-4e23-4952-97a1-dc3ea2c88704
md"#### Separate in train and test subsets
There is data of 57 years, therefore we split the data in the train and test subsests in the following manner: |train| = 45 first years and |test| = 12 last years.  
Also, for simplicity, we will be measuring the populations in 'thousands of animals'."

# ╔═╡ 8ee839c4-881e-40f2-8347-2fd36b1be764
begin
	# Create new data frames, with numbers in Float32 format
	df_train = mapcols(x -> Float32.(x .÷ 1000), rawdata[1:45,[:hare, :lynx]])
	df_test  = mapcols(x -> Float32.(x .÷ 1000), rawdata[46:end,[:hare, :lynx]])
	df = vcat(df_train, df_test)
end

# ╔═╡ 21ea978e-0c12-4eb0-9cf2-feb80f69ddbe
md"#### Visualization of the entire dataset"

# ╔═╡ 7e98f6fc-3b5c-4fab-b0bd-e64c5721ac49
begin
	# Line chart of the data with Makie
	dataset_figure_lines= Figure()
	ax_1 = GLMakie.Axis(dataset_figure_lines[1,1],
		xlabel="Year",
		ylabel="Thousands of Animals")
	GLMakie.lines!(rawdata.year, df.hare, label="Hare population")
	GLMakie.lines!(rawdata.year, df.lynx, label="Lynx population")
    axislegend(ax_1; position=:rt)
	dataset_figure_lines
end

# ╔═╡ e209ccec-72dc-4eb1-92a5-0ee0a5953074
md"## The MLP Model

The built model was developed following the [MLP time series forecasting tutorial](https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/). 
Based on this we developed a *Multiple Parallel Series / Multivariate forecasting MLP model*.
The idea of it is to train a Multi-Layer Perceptron (MLP) for, given multiple parallel time series (say two in our case: `a` and `b`): [(a1,b1), (a2,b2), (a3,b3),..., (an, bn)], to be able to predict the next value for each of the time series: (an+1, bn+1).

For doing this we can follow two different approaches:
- (1) Predict with 1 single MLP. ~*Assuming a and b are dependent*~
- (2) Predict each output series with a different MLP (1 per series). ~*Assuming a and b are independent*~

In particular, we are going to follow the 1st approach, since we understand that this way the model learns that both time series `a` and `b` are relationed, and that the outputs, although different, depend on a joint input [(a1,b1), (a2,b2), ...,(aw,bw)].
"

# ╔═╡ 3aa4a5cd-3000-4744-8211-56fcb442fd8d
begin
	image_path = "./images/MLP-model-0.png"
	image = load(image_path)
	image 
end

# ╔═╡ e12f5cc2-61e9-491d-9551-1754a8d3fac2
md"### Window-based

Another important aspect that we do in our model is to follow a window based approach. That is, the input of the MLP is a windowed time series of size `w`: `[(a1,b1), (a2,b2), ...,(aw,bw)]`. 

This way, we make that a certain prediction only depends on the last `w` values of the time series. Since *does the number of hares in 1860 matter when trying to predict the number of hares in 1900?* (See the nature of the time series in the `Visualization of the entire dataset` subsection).

E.g. w=3, input:[(a1,b1), (a2,b2), (a3,b3)] and the MLP model will produce as output (a'4,b'4).

Therefore, the challenge here is to select a proper window size `w`.
"

# ╔═╡ 4670ce99-3ca0-4ff9-ab53-c3462a193d28
md"#### Preparing the training data
We create customised train datasets in the windowed input format.  
That is, since what we have is a time series like hares = [h1, h2, h3, ..., hn], to be able to train the model following the window approach we need to format this training data so that we have all the corresponding slices of size `w` in this series and its corresponding target output value to train/fit the model.

Example: 
- `w` = 3
- hares = [4, 2, 1, 5, 7, 6]
- lynxes = [2, 3, 4, 3, 1, 3]

For each time series we will build all the corresponding input slices of size `w` with the corresponding target output (the next value of that slice):
- hares: [4,2,1]:5, [2,1,5]:7, [1,5,7]:6
- lynxes: [2,3,4]:3, [3,4,3]:1, [4,3,1]:3

As inputs we will provide $[h_{i-3}, h_{i-2}, h_{i-1}, l_{i-3}, l_{i-2}, l_{i-1}]$. As output, similarly we will expect to get two values $[h_i, l_i]$.

	- input_1 = [4,2,1]+[2,3,4] = [4,2,1,2,3,4] output_1 = [5]+[3] = [5,3] 
    - input_2 = [2,1,5]+[3,4,3] = [2,1,5,3,4,3] output_2 = [7]+[1] = [7,1]
	- input_3 = [1,5,7]+[4,3,1] = [1,5,7,4,3,1] output_2 = [6]+[3] = [6,3]

Therefore, the MLP will learn its weights and biases based on this windowed approach training, so that the MLP model will try to adjust to the output value of each time series whenever the input is like the given one.
"

# ╔═╡ 05c79af6-d9bb-4804-ac4d-b7398500e318
# Function to split the training data in window-based manner 
# window size w = lookback
function create_dataset(data, lookback)
    X, Y = [], []
    for i in lookback+1:length(data)
        push!(X, data[i-lookback:i-1])
        push!(Y, data[i])
    end
    return hcat(X...), vcat(Y...)'
end

# ╔═╡ 3d0bf36d-caa6-4122-8ae1-dafba79b1886
md"Looking at the nature of the time series in the `Visualization of the entire dataset` subsection, we select 3~5 as window size. Greater or smaller window sizes could also be selected."

# ╔═╡ 42194c0b-0329-48ca-b0ba-8b14c38367de
lookback = 5  # Window size 

# ╔═╡ ade309d5-05f7-4dbf-8496-c8f66912d1d5
X_a, Y_a = create_dataset(df_train.hare, lookback)

# ╔═╡ 96eafc94-1f06-4623-878f-308fbde6cc64
X_b, Y_b = create_dataset(df_train.lynx, lookback)

# ╔═╡ da78741c-d3f5-44be-8798-662c9de8b74b
md"
**In Flux, by default, each column is treated as a separate data point in matrix inputs, so the target data is also a matrix with the same setup.**"

# ╔═╡ 133e7a11-ef3b-473a-9900-561d537ab9d4
# as before but now concatenating the b series corresponding column input for each of the columns -> vertical concatenation 
X_ab = vcat(X_a, X_b)

# ╔═╡ 67a153de-d4ce-4e64-a426-eba86551c177
Y_ab = vcat(Y_a, Y_b)

# ╔═╡ d44f537b-264d-4a5d-985a-f246ca12b75b
md"
#### Training the model
*train!(loss, params, data, opt; cb)* 

*Flux train function*:
For each datapoint `d` in data, compute the gradient of `loss` with respect to params through backpropagation and call the optimizer `opt`. If d is a tuple of arguments to loss call loss(d...), else call loss(d). 
A callback is given with the keyword argument `cb`.

- `loss`: loss function -> to evaluate how well the model is performing and do the adjust of the weights and biases accordingly.
- `params(model)`: where the parameters of the model to be adjusted to minimize the loss (weights and biases) are expressed.
- `data`: training data, with inputs and target outputs, for each of the inputs:(`X_ab`,`Y_ab`)
- `optimizer`: GD (Gradient Descent), SGD (Stochastic Gradient Descent), ADAM (ADaptive Moment Estimation)...
"

# ╔═╡ 96017ab0-367b-49fe-9ebe-9963aca2cbf8
md"**1. MLP definition** 

*Chain -> to stack layers, Dense -> fully connected neural network layers*

- 1st layer: Dense(`lookback * 2`, `hiddenNeurons`, `relu`) -- it is the hidden layer
  - Input size: `lookback * 2` (window size = lookback), so `#lookback` inputs per time series, 2 in this case
  - Output size: `hiddenNeurons` -> hidden layer has `#hiddenNeurons` neurons
  - Activation function: ReLU - introduces non linearity to the model

- Output layer: Dense(`hiddenNeurons`,2)
  - Input size: `hiddenNeurons`, to match the outputs of all the neurons in the hidden layer (`hiddenNeurons`)
  - Output size: 2, since we are forecasting a single future value for each of the two time series
"

# ╔═╡ d06601ff-99f8-4696-a2c5-5d2c1d493275
begin
	image_path_1 = "./images/MLP-model-1.png"
	image1 = load(image_path_1)
	image1 
end

# ╔═╡ 49884347-9130-40cc-a845-57da66e85718
# Number of neurons in the hidden layer
hiddenNeurons = 15

# ╔═╡ 37e9059f-f0fa-4c22-a707-37cc68b23419
# model
model_ab = Flux.Chain(
	Flux.Dense(lookback*2 => hiddenNeurons, relu; init = Flux.glorot_uniform(MersenneTwister(1000))),
	Flux.Dense(hiddenNeurons => 2, init = Flux.glorot_uniform(MersenneTwister(1000)))
model_ab = Flux.Chain(
	Flux.Dense(lookback*2, hiddenNeurons, relu),
	Flux.Dense(hiddenNeurons,2)
)

# ╔═╡ 92dfde86-f396-4f1d-a54a-83b6aeb725f0
ps_ab = Flux.params(model_ab)

# ╔═╡ bb8b1da2-2fe1-4657-a473-8d19e0bb3e22
md"**2. Definition of the loss function**

MSE between the model predicted values (`ŷ1`, `ŷ2` = model(x)) and the target values: `y1, y2`.
"

# ╔═╡ 6753be48-af78-42aa-9f0d-028498a3bcd3
md"**3. Optimizer**  

- initially: simple Gradient Descent
- More advanced optimizers: ADAM (ADaptive Moment Estimation)"

# ╔═╡ 81812092-33a3-49a2-89ed-6b79ccc685bc
#optimizer_ab = Descent(0.01)
optimizer_ab = ADAM(0.01)

# ╔═╡ 38b3a708-4bfc-44c7-805f-82d505f8ced3
data_pair_ab = [(X_ab,Y_ab)]

# ╔═╡ 72d63b54-1b2f-4aa3-8e18-acad328ed006
md"**Training loop**"

# ╔═╡ 1fb5ba25-e9bd-418c-8606-52db645bf290
#=╠═╡
begin
		# TOTRY: Variate the # of epochs
		epochs = 500
		for epoch in 1:epochs
			Flux.train!(loss_ab, ps_ab, data_pair_ab, optimizer_ab)
			println("Epoch $epoch, Loss: $(loss_ab(X_ab,Y_ab))")
		end
end
  ╠═╡ =#

# ╔═╡ e1ce5c16-c38a-47d9-a78a-a0e8552b0178
md"#### Testing the model

We are going to build the predictions of the test values, for 🐇 and 🐆, with the constructed model and then check the error obtained in terms of the difference with the real test values for those years. 

The predictions obtained by the model are going to be constructed in the `pred_test_a` and `pred_test_b` vectors for the hares and lynxes, respectively.

The construction of the prediction vectors will **not** be done using the real test data values, instead we will make use of the self-predicted values by the model.

For instance, if we assume the window size `lookback=3`, we will construct the test years predictions of a certain time series like:

- `[x_43, x_44, x_45] -> x_46'` (*`x_43,x_44,x_45` are real values from the train dataset*)
- `[x_44, x_45, x_46'] -> x_47'` (*however, `x_46'` is not the real test value, but the one that has just been predicted by the model in the previous step*).
- `[x_45, x_46', x_47'] -> x_48'`
- `[x_46', x_47', x_48'] -> x_49'`
- ...
- `[x_54', x_55', x_56'] -> x_57'`

Once we have this predicted ``|\hat{test}|`` output set, we compare it against the real
``|test|`` dataset to obtain the error and measure the performance.
"

# ╔═╡ 03fa0548-3ad4-43f7-a8c8-be73402e8879
begin
	test_data_a = df_test.hare
	test_data_b = df_test.lynx
	pred_test_a = Float32[]
	pred_test_b = Float32[]
	input_vec_a = Float32[]
	input_vec_b = Float32[]
	# fill until the w (lookback) size, initially all from the last w training data elements
	for i in length(df_train.hare)-lookback+1:length(df_train.hare)
		push!(input_vec_a, df_train.hare[i])
		push!(input_vec_b, df_train.lynx[i])
	end
end

# ╔═╡ 3a745c16-4dad-4b23-b12e-4a2d45d8c061
for i in 1:length(test_data_a)
	# put together the two vectors to do the prediction
	input_vec_ab = vcat(input_vec_a, input_vec_b)
	output_ab = model_ab(input_vec_ab)
	# The first predicted value is for the a series and the second for the b series
	push!(pred_test_a, output_ab[1])
	push!(pred_test_b, output_ab[2])
	# prepare the new following input vector (size of this vector = lookback / window size)
	for w in 1:lookback-1
		input_vec_a[w] = input_vec_a[w+1]
		input_vec_b[w] = input_vec_b[w+1]
	end
	input_vec_a[lookback] = output_ab[1]
	input_vec_b[lookback] = output_ab[2]
end

# ╔═╡ 7ac03a08-0ff8-4dfd-9766-d12f12de75ae
md"**Joint error - MSE(hare) + MSE(lynx)** = $(Flux.Losses.mse(test_data_a, pred_test_a) + Flux.Losses.mse(test_data_b, pred_test_b))"

# ╔═╡ 932d8ef8-5a3c-4a0e-939f-30a059fae242
md"#### Hares

MSE = $(Flux.Losses.mse(test_data_a, pred_test_a))"

# ╔═╡ 61926b15-da07-49fc-a488-3d135707b882
begin
	x_axis = rawdata.year[end-11:end]
	fig_a = Figure()
	ax_a = GLMakie.Axis(fig_a[1, 1], xlabel="Year",
		ylabel="Thousands of Animals", title="Hares test results")
	GLMakie.lines!(ax_a, x_axis, test_data_a, label="Real hare population")
	GLMakie.lines!(ax_a, x_axis, pred_test_a, label="Predicted hare population")
	axislegend(ax_a; position=:rt)
	fig_a
end

# ╔═╡ 3a68235c-3049-4f5e-8cbb-a744661ba914
md"#### Lynx
MSE = $(Flux.Losses.mse(test_data_b, pred_test_b))
"

# ╔═╡ 8a074363-d831-44c1-8854-9142c1e02cae
begin
	fig_b = Figure()
	ax_b = GLMakie.Axis(fig_b[1, 1], xlabel="Year",
		ylabel="Thousands of Animals", title="Lynx test results")
	GLMakie.lines!(ax_b, x_axis, test_data_b, label="Real lynx population")
	GLMakie.lines!(ax_b, x_axis, pred_test_b, label="Predicted lynx population")
	axislegend(ax_b; position=:rt)
	fig_b
end

# ╔═╡ d25b8844-1d46-4c0f-9159-feeb664d2865
md"
| #epochs 	| Optimizer (learning rate) 	| Window size 	| hiddenNeurons 	| MSE 	|
|:---:	|:---:	|:---:	|:---:	|:---:	|
| 1000 	| Gradient_Descent(0.01) 	| 3 	| 15 	| 1347 + 266 = 1613   (BAD - GD gives  a straight line as prediction!) 	|
| 100 	| ADAM(0.01) 	| 3 	| 15 	| 1098 + 547 = 1645  (underfitting - flat shape curve) 	|
| 1000 	| ADAM(0.01) 	| 3 	| 15 	| 1384 + 193 = 1577 	|
| 500 	| ADAM(0.01) 	| 3 	| 15 	| 895 + 464 = 1359 	|
| 500 	| ADAM(0.01) 	| 10 	| 150 	| 5997 + 2557 = 8555   (overfitted - error ~ 0 for train dataset) 	|
| 500 	| ADAM(0.01) 	| 10 	| 50 	| 7178 + 4435 = 11613   (overfitted - error ~ 10^-2 for train dataset) 	|
| 500 	| ADAM(0.01) 	| 5 	| 50 	| 1353 + 486 = 1839   (overfitted - error ~ 10^-2 for train dataset) 	|
| 500 	| ADAM(0.01) 	| 5 	| 15 	| 276 + 310 = 587 	|
| 100 	| ADAM(0.01) 	| 5 	| 15 	| 759 + 617 = 1377 	|
| 200 	| ADAM(0.01) 	| 3 	| 15 	| 1110 + 561 = 1672 	|
| 200 	| ADAM(0.01) 	| 3 	| 5 	| 972 + 448 = 1420  (quite flat shape of the curve) 	|
| 200 	| ADAM(0.01) 	| 3 	| 25 	| 771 + 538 = 1310  	|
| 500 	| ADAM(0.01) 	| 3 	| 25 	| 406 + 431 = 837 	|
"

# ╔═╡ 8bd50f06-8c84-4f26-ba0d-aebb4fec0898
md"
As described in our prediction model there are many tunnable paramters. 
The tunning process to select a combination that provides a good enough model in
terms of the prediction of the test dataset is described in the previous table. 
From the experiments done we derived the following conclusions in relation with the different parameters:

- **Optimizer**: ADAM (ADaptive Moment Estimation) is preferrable over GD (Gradient Descent), since for the last many more epochs of training were needed to produce similar MSEs, also GD tended to produce straight lines as prediction, which of course are far to match the periodicity seen in our data.

- **Window size** 🪟: 3~5 was the preferred size, since higher window sizes tended to show overfitting in the training dataset, and the model was not able to generalize for the prediction of the testing values. Intuituively, small window sizes are better due to the nature of our time series periodicity (see it in `Visualization of the entire dataset` subsection).

- **HiddenNeurons**: In relation with the neurons on the hidden layer of the MLP, it was observed that a high number tended to produce overfitting on the training data whereas a low number tended to produce underfitting reflected in a quite flat shape of the prediction curve. Therefore an \"intermediate\" number (e.g. 15~30) was selected.

- **`#epochs`**: The behavior in this case was similar as the described for the `hiddenNeurons`, meaning that a high number of epochs tended to produce an overfitted model and a low number of epochs tended to produce an underfitted model. Therefore, the preferred number of epochs was selected to be ~500.

So far, with these experiments we managed to obtain a total MSE (`MSE(hares) + MSE(lynx)`)
of `587`, with: `#epochs = 500`, `optimizer = ADAM(0.01)`, `window = 5` and 
`hiddenNeurons = 15`.
"

# ╔═╡ 97532497-d321-4d5f-bd7e-89c349ae16da
md"""
# Lokta-Volterra system
"""

# ╔═╡ 899d5be3-7d86-405c-99ca-6e17af53214a
md"""
We wish to model the evolution of our twin time series, $x \sim 🐇$ and $y \sim 🐆$, as a Lokta-Volterra dynamic system, which is defined as follows:

$\left\{ \begin{align}
x'(t) &= \alpha x(t) - \beta x(t)  y(t)\\
y'(t) &= -\gamma y(t) + \delta x(t)  y(t)
\end{align} \right.$
"""

# ╔═╡ 7f7f5196-8464-46c7-a84d-55bde0e527a8
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# ╔═╡ a0d8b637-99d8-47d3-9b16-bb570ff30a77
md"""

For our case, we may impose that the initial time $t=0$ corresponds to the first year in our train dataset, namely $(rawdata.year[1]). The final time will be ``t=`` $(size(df_train)[1]-1), corresponding to the year $(rawdata.year[size(df_train)[1]]).

Therefore, the initial conditions for the system are ``x(0) = `` $(Integer(df.hare[1])) and ``y(0) = `` $(Integer(df.lynx[1])).
"""

# ╔═╡ 46efb936-c62f-4686-a045-0f03e503a1f8
begin
	# Simulation interval
	tspan = Float32.((0.0, size(df_train)[1]-1))
	
	# Initial condition
	u0 = [df_train.hare[1], df_train.lynx[1]]
	
	# Initial values for the parameters
	p = [ 1.0777069248324076, 0.034059726645187034, 1.040486193025437 ,0.028715996648942803] # [α, β, δ, γ]
end

# ╔═╡ 79427815-d276-4169-aa18-8e97a5e30775
md"""
With Pluto.jl, we can get some interactivity to test out how the values for the parameters affect the final solution!

α= $(@bind a Scrubbable(p[1])) β= $(@bind b Scrubbable(p[2])) δ= $(@bind c Scrubbable(p[3])) γ=a $(@bind d Scrubbable(p[4]))
"""

# ╔═╡ e0941276-91bf-4aca-90df-87c1bd3ac6ae
# Symbolic solver of the ODE system
begin
	# Define our state variables: state(t) = initial condition
	@variables t x(t)=df.hare[1] y(t)=df.lynx[1]
	
	# Define our parameters
	@parameters α=a β=b δ=c γ=d
	
	# Define our differential: takes the derivative with respect to `t`
	D = Differential(t)
	
	# Define the differential equations
	eqs = [D(x) ~ α * x - β * x * y
	       D(y) ~ -δ * y + γ * x * y]
	
	# Bring these pieces together into an ODESystem with independent variable t
	@named sys = ODESystem(eqs, t)
	
	# Symbolically Simplify the System
	simpsys = structural_simplify(sys)
	
	# Convert from a symbolic to a numerical problem to simulate
	interactive_prob = ODEProblem(simpsys, [], tspan)
	interactive_sol = solve(interactive_prob);

	model_figure = Figure()
	ax2 = GLMakie.Axis(model_figure[1,1])
	GLMakie.ylims!(ax2, 0, 150)
	GLMakie.lines!(ax2, interactive_sol.t, interactive_sol[1,:], label="hares")
	GLMakie.lines!(ax2, interactive_sol.t, interactive_sol[2,:], label="lynx")
    axislegend(ax2; position=:rt)
	model_figure
	
end

# ╔═╡ 85c89222-0578-40c6-9394-37b0c3f18d30
md"""
Do note that for some values of the parameters, there will be times when either population get very close to zero. This can lead to the differential equation being numerically unstable (aka _stiff_).

In the future, we will default to using ODE solvers designed for treating stifness.
"""

# ╔═╡ cc725260-e780-495e-817d-6950622949ef
md"""
First, we need to define a criteria for _fitness_, a **loss function**.


``$\mathcal{L}( \alpha, \beta, \delta, \gamma )= \sum_{i=1}^{57}
\Vert
	(\hat{x}_i, \hat{y}_i) - (x(t_i), y(t_i))
\Vert ^2_2 $``

Where $\hat{x}, \hat{y}$ represent our date points and $x(t), y(t)$ are the solutions to the Lokta-Volterra systems with the given parameters.
"""

# ╔═╡ fa4a629b-28f5-4007-8c8a-9682497c4d72
md"""
# Neural ODE 
"""

# ╔═╡ 78b0e77b-1746-4528-bb08-b761ef088c84
begin
	u_train = transpose(Matrix(df_train))
	t_train = Float32.(Array(0.0:Float32(size(df_train)[1]-1)))
	train_years = rawdata.year[1:45]
	rng = Random.MersenneTwister(1000)	
end

# ╔═╡ 01e51c71-ad3b-4f82-924b-edd8d0232e11
function neural_ode(t, data_dim)
    f = Lux.Chain(
        Lux.Dense(data_dim, 32, Lux.swish),
        Lux.Dense(32, 64, Lux.swish),
        Lux.Dense(64, data_dim)
    )

    node = NeuralODE(
        f, extrema(t), Tsit5(),
        saveat=t,
        abstol=1e-9, reltol=1e-9
    )
    p, state = Lux.setup(rng, f)

    return node, ComponentArray(p), state
end

# ╔═╡ 62557d5e-5d63-4f9b-af67-2df2ae01c28d
predict(y0, t, p=nothing, state=nothing) = begin
    node, p_random, state_random = neural_ode(t, length(y0))
    if p === nothing p = p_random end
    if state === nothing state = state_random end
    ŷ = Array(node(y0, p, state)[1])
end

# ╔═╡ 44ff87d8-50df-41c4-a3b4-af96e2e29438
function plot_trajectories(y, pred)
    n = size(y, 2)
    m = size(pred,2)
    years = rawdata.year
	fig_node = Figure()
	ax3 = GLMakie.Axis(fig_node[1,1], xlabel="Year", ylabel="Thousands of Animals",
	ylim = (0,150))
	
    GLMakie.series!(ax3, pred, labels=["x(t)" "y(t)"])
    GLMakie.scatter!(plt, years[1:n], y[1, :]; 
    label = "hare", color=:green)
    GLMakie.scatter!(plt, years[1:n], y[2, :];
    label = "lynx", color=:red2)
    axislegend(ax3; position=:rt)
    display(plt)
	return fig_node
end

# ╔═╡ f4017dce-3179-45d1-bc9e-7a367d637114
function train_one_round(node, p, st, y, maxiters, lr)
    pred(p) = Array(node(u0, p, st)[1])
    loss(p) = sum(abs2, pred(p) .- y)/length(u_train)
    callback(p, l) = begin
        println(l)
        plot_trajectories(y, pred(p))
        return false
    end
    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((p, _ ) -> loss(p), adtype)
    optprob = OptimizationProblem(optf, p)
    callback(p, loss(p))
    res = solve(optprob, Optimisers.ADAMW(lr), maxiters=maxiters)
    return res.u, st, pred(p)
end


# ╔═╡ 6e994f72-4dd4-43c1-805c-2d1d2dddc66e
function train(y, t, maxiters = 150, lr = 1e-2)
    p=nothing
    state=nothing
    
    for k in 3:3:length(t_train)
        println("Training batch of first $k values")
        node, p_new, state_new = neural_ode(t[1:k], size(y, 1))
        if p === nothing p = p_new end
        if state === nothing state = state_new end

        p, state, pred = train_one_round( node, p, state, y[:,1:k], maxiters, lr)
        plot_trajectories(y, pred)
    end
    p, state
end

# ╔═╡ 021e10d2-438c-48f4-8034-c851b36a5a56
t_train

# ╔═╡ 9fc1eddb-fbf3-4034-a0f7-b4514ee3f13c
prediction = predict(u0, t_train)

# ╔═╡ 14b92513-f2ab-4d60-9772-eebbdce50f99
plot_trajectories(u_train, prediction)

# ╔═╡ 1e8fa51a-cdc2-41da-a6e0-23a368fcf479
res = train(u_train, t_train)

# ╔═╡ de81ad4e-7fb1-4fa0-8924-4b612e1b5821
md"""
# Direct Optimiziation
"""

# ╔═╡ ff35d4cf-3a80-4dda-92f5-cff14ecd9016
# ╠═╡ disabled = true
#=╠═╡
pguess = [1.0, 0.1, 1.0 ,0.1]
  ╠═╡ =#

# ╔═╡ 54b59818-0a5d-4964-bf65-696a06966123
prob_neuralode = NeuralODE(NN, tspan, Tsit5(); saveat = 1)

# ╔═╡ 75dbd942-0305-48ba-8a32-55148b9d700b
function predict_neuralode(params)
    Array(prob_neuralode(u0, params, state)[1])
end

# ╔═╡ 811af0da-0077-4ca0-8f86-4a1712dd48b0
md"""
Switch to different optimizer lol
"""

# ╔═╡ 3e94f5b7-5c00-4eb5-91d4-890be641b0b5
plot_trajectory( predict_neuralode(result_neuralode2.u))

# ╔═╡ f8e99383-f22b-4cf5-97f1-304a3c901883
# ╠═╡ disabled = true
#=╠═╡
result_ode
  ╠═╡ =#

# ╔═╡ 5a3e7c61-0837-476f-9488-cd553ca566c4
length(result_neuralode2.u.layer_1[:])

# ╔═╡ 2d69e051-7651-4e37-9f00-ff5ba23ee74a
function plot_trajectories_old(y, pred)
	fig_node = Figure()
	ax3 = GLMakie.Axis(fig_node[1,1],
			xlabel="Year",
			ylabel="Thousands of Animals")
	tsteps = range(tspan[1]+1, tspan[2]+1; step=1)
	empty!(ax3)
	GLMakie.series!(ax3, pred, labels=["Hare", "Lynx"], color=[:blue, :orange])
	GLMakie.scatter!(ax3, tsteps, true_values[1,:], color = :darkblue)
	GLMakie.scatter!(ax3, tsteps, true_values[2,:], color = :chocolate)
	axislegend(ax3; position=:rt)
	return fig_node
end

# ╔═╡ 6b550704-5e9c-4bfd-9b49-239029b789b3
#=╠═╡
loss(pguess)
  ╠═╡ =#

# ╔═╡ 90b70812-4f93-4950-b253-60d3b4d09949
# ╠═╡ disabled = true
#=╠═╡
callback = function (p, l, sol)
	# Tell Optimization.solve to not halt the optimization. If return true, then
	# optimization stops.
	if l == Inf 
		@info "el solver falló. Terminating..."
		return true
	else
	    println(l)
	    plt = Plots.plot(sol, ylim = (0, 150), label = "Current Prediction")
	    Plots.scatter!(plt, true_values, label = "data")
		return false
	end
    
end
  ╠═╡ =#

# ╔═╡ 4addce6d-7c13-485e-9c06-7c108d8a17bb
md"""
# Conclusions
The MLP model was successfull in the series forecast, and quite easy to implement compared to the ODE models.

From the beggining, we were aware that it is impossible to find values for the parameters (α , β, δ, γ) producing Lokta-Volterra ODE perfectly fitting our data. What we did not expect is that the potential _stiffness_ of ODE would end up breaking the Optimization scheme.

Our NeuralODE model was able to learn the curves when it was fed a a small quantity of data, but performed rather disappointingly when the training data was extended beyond *. 
We believe that the issue comes from the fact that NeuralODE do not (by default) hold the same Universal Approximation Properties as Neural Networks. This has been studied by several authors, the latest work to our knowledge is by [Teshima et. al.](https://arxiv.org/abs/2012.02414).

## Future work

### Augmentation

A clear intuition on why that is the case can be found on the image below, representing a one-dimensional NeuralODE h'(t) = f(h(t), t). It is impossible to find a continuous vector field f that would make any pair of trajectories intersect.
"""

# ╔═╡ 1f480571-7bf0-44d1-bf70-6e7a3c4d08c0
load("./images/node-non-universality.png")

# ╔═╡ 2910663e-c29b-4df6-a0cc-18f9862a4484
md"""
The simplest solution to this problem was proposed by [Dupont et. al.](https://arxiv.org/abs/1904.01681). One can _augment_ the dimension of the state, giving enough space to the trajectories to cross without intersecting. This idea is actually very similar to kernel feature spaces.

The image belows shows an example of this for a classification task, where NeuralODEs try to find trajectories separating two levels of data points.
"""

# ╔═╡ e6a97d15-b9d0-4377-afd1-98861f89f9a0
load("./images/augment.png")

# ╔═╡ b730c32f-dcd6-4b32-ab0f-b3a0cd6fcf46
md"""
We will be implementing that in the future.

### Hybrid Model

We plan on implementing a model that combines the NeuralODE with the direct ODE parameter optimization. In this case, the Neural Network will represent the dynammics of the difference between a pure Lokta-Volterra ODE and our 'noisy' dataset. 

$\left\{ \begin{align}
x'(t) &= \alpha x(t) - \beta x(t)  y(t) + NN_1(x, y, \theta)\\
y'(t) &= -\gamma y(t) + \delta x(t)  y(t) + NN_2(x, y, \theta)
\end{align} \right.$

Our training scheme will try to learn both $\alpha, \beta, \gamma, \delta$ and the Neural Network parameters $\theta$.

We believe that model will be able to approximate the trajectory, as it has already been studied by [Rackauckas et.al](https://arxiv.org/abs/2001.04385) in their paper "Universal Differential Equations for Machine Learning". 

We have already tried to implement it, but we faced some issues. Our goal is to fix the bugs and present the model for the following delivery.
"""

# ╔═╡ a69aaafa-ac8e-4031-b997-8e9c227e3987
begin
	true_values = transpose(Matrix(df))
	function loss(newp)
	    newprob = remake(prob, p = newp)
	    sol = solve(newprob, Rosenbrock23(autodiff=false), saveat = 1)
		
	    loss = try sum(abs2, sol .- true_values) 
		catch e
				return Inf, sol
		end
		return loss, sol  
	end
end

# ╔═╡ 1fcb4488-b46a-47cf-83b8-85a97e052423
#=╠═╡
function loss_ab(x,y) 
	pred = model_ab(x)
	return mean((pred .- y).^2)
end
  ╠═╡ =#

# ╔═╡ 38ed433a-b084-403a-bc2f-0890a7b52353
#=╠═╡
begin
	adtype = Optimization.AutoZygote()
	
	optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
	optprob = Optimization.OptimizationProblem(optf, pinit)
	
end


  ╠═╡ =#

# ╔═╡ 7c37627b-f771-4084-9163-08e361723bbc
# ╠═╡ disabled = true
#=╠═╡
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
  ╠═╡ =#

# ╔═╡ 7d11f193-1c22-41d7-a29d-d6c5a1051b59
# ╠═╡ disabled = true
#=╠═╡
adtype = Optimization.AutoForwardDiff()
  ╠═╡ =#

# ╔═╡ 7e35d18a-ad8a-497a-aeca-a32e2cce16f4
# ╠═╡ disabled = true
#=╠═╡
loss_ab(x,y) = Flux.Losses.mse(model_ab(x), y) 
  ╠═╡ =#

# ╔═╡ 2112b7df-4470-4748-9334-a77fb0cbc119
begin
	true_values = transpose(Matrix(df_train))
	function loss_neuralode(p)
	    pred = predict_neuralode(p)
	    loss = sum(abs2, true_values .- pred)
	    return loss, pred
	end
end

# ╔═╡ Cell order:
# ╠═533f45ff-47c9-479d-8e4c-c81c33cc4e0f
# ╟─d0e5a39f-2da0-410f-a9be-009b039ec891
# ╟─babeac0c-aac9-4725-88b7-6a8d9008d6c0
# ╟─a3b24f11-5102-47a4-bad2-ac6b50d78a7b
# ╟─a9b0ae89-13d4-48b3-8e98-e9a1a714b0b4
# ╟─5aa263f7-fb27-43f2-abe9-733140d8a0ca
# ╠═e920cd2f-c996-4e67-81ab-5e98fb17d386
# ╟─336d223c-4e23-4952-97a1-dc3ea2c88704
# ╠═8ee839c4-881e-40f2-8347-2fd36b1be764
# ╟─21ea978e-0c12-4eb0-9cf2-feb80f69ddbe
# ╠═7e98f6fc-3b5c-4fab-b0bd-e64c5721ac49
# ╟─e209ccec-72dc-4eb1-92a5-0ee0a5953074
# ╠═3aa4a5cd-3000-4744-8211-56fcb442fd8d
# ╟─e12f5cc2-61e9-491d-9551-1754a8d3fac2
# ╟─4670ce99-3ca0-4ff9-ab53-c3462a193d28
# ╠═05c79af6-d9bb-4804-ac4d-b7398500e318
# ╟─3d0bf36d-caa6-4122-8ae1-dafba79b1886
# ╠═42194c0b-0329-48ca-b0ba-8b14c38367de
# ╟─ade309d5-05f7-4dbf-8496-c8f66912d1d5
# ╠═96eafc94-1f06-4623-878f-308fbde6cc64
# ╟─da78741c-d3f5-44be-8798-662c9de8b74b
# ╠═133e7a11-ef3b-473a-9900-561d537ab9d4
# ╠═67a153de-d4ce-4e64-a426-eba86551c177
# ╟─d44f537b-264d-4a5d-985a-f246ca12b75b
# ╟─96017ab0-367b-49fe-9ebe-9963aca2cbf8
# ╟─d06601ff-99f8-4696-a2c5-5d2c1d493275
# ╠═49884347-9130-40cc-a845-57da66e85718
# ╠═37e9059f-f0fa-4c22-a707-37cc68b23419
# ╠═92dfde86-f396-4f1d-a54a-83b6aeb725f0
# ╟─bb8b1da2-2fe1-4657-a473-8d19e0bb3e22
# ╠═7e35d18a-ad8a-497a-aeca-a32e2cce16f4
# ╠═1fcb4488-b46a-47cf-83b8-85a97e052423
# ╟─6753be48-af78-42aa-9f0d-028498a3bcd3
# ╠═81812092-33a3-49a2-89ed-6b79ccc685bc
# ╠═38b3a708-4bfc-44c7-805f-82d505f8ced3
# ╟─72d63b54-1b2f-4aa3-8e18-acad328ed006
# ╠═1fb5ba25-e9bd-418c-8606-52db645bf290
# ╟─e1ce5c16-c38a-47d9-a78a-a0e8552b0178
# ╠═03fa0548-3ad4-43f7-a8c8-be73402e8879
# ╠═3a745c16-4dad-4b23-b12e-4a2d45d8c061
# ╟─7ac03a08-0ff8-4dfd-9766-d12f12de75ae
# ╟─932d8ef8-5a3c-4a0e-939f-30a059fae242
# ╟─61926b15-da07-49fc-a488-3d135707b882
# ╟─3a68235c-3049-4f5e-8cbb-a744661ba914
# ╟─8a074363-d831-44c1-8854-9142c1e02cae
# ╟─d25b8844-1d46-4c0f-9159-feeb664d2865
# ╟─8bd50f06-8c84-4f26-ba0d-aebb4fec0898
# ╟─97532497-d321-4d5f-bd7e-89c349ae16da
# ╟─899d5be3-7d86-405c-99ca-6e17af53214a
# ╠═7f7f5196-8464-46c7-a84d-55bde0e527a8
# ╟─a0d8b637-99d8-47d3-9b16-bb570ff30a77
# ╠═46efb936-c62f-4686-a045-0f03e503a1f8
# ╠═79427815-d276-4169-aa18-8e97a5e30775
# ╟─e0941276-91bf-4aca-90df-87c1bd3ac6ae
# ╟─85c89222-0578-40c6-9394-37b0c3f18d30
# ╠═75dbd942-0305-48ba-8a32-55148b9d700b
# ╠═cc725260-e780-495e-817d-6950622949ef
# ╟─fa4a629b-28f5-4007-8c8a-9682497c4d72
# ╠═78b0e77b-1746-4528-bb08-b761ef088c84
# ╠═01e51c71-ad3b-4f82-924b-edd8d0232e11
# ╠═f4017dce-3179-45d1-bc9e-7a367d637114
# ╠═6e994f72-4dd4-43c1-805c-2d1d2dddc66e
# ╠═62557d5e-5d63-4f9b-af67-2df2ae01c28d
# ╠═44ff87d8-50df-41c4-a3b4-af96e2e29438
# ╠═2d69e051-7651-4e37-9f00-ff5ba23ee74a
# ╠═021e10d2-438c-48f4-8034-c851b36a5a56
# ╠═9fc1eddb-fbf3-4034-a0f7-b4514ee3f13c
# ╠═14b92513-f2ab-4d60-9772-eebbdce50f99
# ╠═1e8fa51a-cdc2-41da-a6e0-23a368fcf479
# ╠═de81ad4e-7fb1-4fa0-8924-4b612e1b5821
# ╠═ff35d4cf-3a80-4dda-92f5-cff14ecd9016
# ╠═54b59818-0a5d-4964-bf65-696a06966123
# ╠═811af0da-0077-4ca0-8f86-4a1712dd48b0
# ╠═7d11f193-1c22-41d7-a29d-d6c5a1051b59
# ╠═3e94f5b7-5c00-4eb5-91d4-890be641b0b5
# ╠═38ed433a-b084-403a-bc2f-0890a7b52353
# ╠═6b550704-5e9c-4bfd-9b49-239029b789b3
# ╠═f8e99383-f22b-4cf5-97f1-304a3c901883
# ╠═2112b7df-4470-4748-9334-a77fb0cbc119
# ╠═7c37627b-f771-4084-9163-08e361723bbc
# ╠═5a3e7c61-0837-476f-9488-cd553ca566c4
# ╠═a69aaafa-ac8e-4031-b997-8e9c227e3987
# ╠═90b70812-4f93-4950-b253-60d3b4d09949
# ╟─4addce6d-7c13-485e-9c06-7c108d8a17bb
# ╟─1f480571-7bf0-44d1-bf70-6e7a3c4d08c0
# ╟─2910663e-c29b-4df6-a0cc-18f9862a4484
# ╟─e6a97d15-b9d0-4377-afd1-98861f89f9a0
# ╟─b730c32f-dcd6-4b32-ab0f-b3a0cd6fcf46
