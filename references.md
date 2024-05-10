

Idea:

Coger distintos modelos de ajuste de funciones / modelos de predicción:  
- 1. Modelo Neural ODEs  
- 2. Modelo full ML, full dependiente de los datos (que ya esté por ahí hecho y 
que sea fácil y poder explicarlo fácilmente - que sea bueno para ajuste de 
series temporales por ejemplo...)  

aplicarlos sobre funciones sencillas / datos sintéticos sencillos para probar el 
ajuste de una forma inicial sencilla.

Cronograma de entrega:

- 1) Explicación e implementación básica de los modelos
- 2) Comparativa de modelos
- 3) Hacer una aplicación real - aplicar sobre un ejemplo más complejo


## 1)

Aplicar sobre el caso básico de 
Modelo ODE:
- https://docs.sciml.ai/Overview/stable/getting_started/first_simulation/#first_sim

Modelo NN:
- MLP (MultiLayer Perceptron)
- Dar mismo # iteraciones / tiempo a ambos modelos
- Comparar en base a métricas de error la f. aproximada obtenida

### 1.A) Temporal model

- A) [Tutorial on adjusting time series](https://datasciencejuliahackers.com/time-series.html):
without NN, with linear mathematical methods.

- A.1) [Forecast.jl - Autoregressive models](https://docs.juliahub.com/Forecast/AiZvp/0.1.0/man/examples/quakes/):
    - Forecast the variable of interest using a linear combination of past values of 
    the variable itself again.
    - More on autoregressive models: [Forecasting tutorial - AR models](https://otexts.com/fpp2/AR.html)

- B) [Time series prediction (in Julia) with NNs](https://medium.com/analytics-vidhya/time-series-prediction-feat-introduction-of-julia-78ed6897910c)

- C) [Tensorflow - Time series forecasting tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series#setup): 
    - See the forecast for 1 feature, multiple time steps...
    - RNNs (Recurrent Neural Networks) good for the task.

- D) [Forecasting tutorial - Neural Network models](https://otexts.com/fpp2/nnetar.html):


#### [FluxArchitectures.jl](https://github.com/sdobber/FluxArchitectures.jl) 

Collection of advanced network architectures for time series forecasting.



More interesting references:

- [Time series prediction](https://juliadynamics.github.io/TimeseriesPrediction.jl/dev/)

- [TSML (Time Series Machine Learning)](https://juliacon.github.io/proceedings-papers/jcon.00051/10.21105.jcon.00051.pdf) - a machine learning package for Time Series in Julia.


________________________________________________________________________________
### 1.B) Predator-Prey model

- [Predator-Prey model](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations): 
wikipedia article, specially well explained and interesting section 
"Biological interpretation and model assumptions".

#### Datasets

- [Paramecium caudatum and Paramecium aurelia](https://rdrr.io/github/adamtclark/gauseR/man/gause_1934_book_app_t04.html#google_vignette) -- used in [this project](https://simiode.org/resources/8557/download/6-067-S-LotkaVolterra-StudentVersion.pdf)
- [Lynx and Hare](https://gist.github.com/michaelosthege/27315631c1aedbe55f5affbccabef1ca)
-- suggested dataset for testing by chatgpt

#### Links to review

- https://www.uv.es/falbe/MatExp/aplicada/modelizacion/Lotka-Volterra/
- https://riull.ull.es/xmlui/bitstream/handle/915/6217/Modelo%20depredador-presa%20de%20Volterra-Lotka.pdf?sequence=1&isAllowed=y
- https://docs.sciml.ai/Overview/stable/getting_started/first_simulation/#first_sim

________________________________________________________________________________


### Related references

- https://research.google/blog/a-neural-weather-model-for-eight-hour-precipitation-forecasting/ 


# MLP

References:   
- https://fluxml.ai/Flux.jl/stable/models/quickstart/  
- [MLP in Julia code](https://github.com/MNoorFawi/multi-layer-perceptron-with-julia)

Seguir con esto:

https://medium.com/@sophb/an-introduction-to-deep-learning-using-flux-part-ii-multi-layer-perceptron-32526b323474

https://fluxml.ai/Flux.jl/stable/models/basics/ 

https://fluxml.ai/Flux.jl/stable/models/quickstart/

https://fluxml.ai/Flux.jl/stable/tutorials/2021-01-26-mlp/

(*) Time series forecasting -> https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/


Okay, here we go, decisions needed to take:

- Architecture: Input layer, hidden layer, output layer

- Loss function

- Activation function: ReLU?

*Slides of Neural Networks - class lecture*
Important notes:
    - Have more observations N (inputs) than regressors h (weights to fit) -->
    *statistical significance of the weights decreases with h and increases with N* (?)

    - Input values: should be preprocessed so that:
        - their mean is zero and their stdev is 1 (use standardization)
        - they are uncorrelated (use PCA)
    - Initial weights: should be small and zero-centered, to avoid initially 
    driving the neurons into saturation.


[MLP time series forecasting - super good tutorial](https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/)

TODO: also explore this: [Configuring a MLP Network for time series forecasting](https://machinelearningmastery.com/exploratory-configuration-multilayer-perceptron-network-time-series-forecasting/)


Based on this, they define:

- Univariate MLP:
Given a time series: [x1, x2, ..., xn] it predicts the next output value (x_n+1)
Single series of observations with a temporal ordering. Model is required
to learn from the series of past observations to predict the next value in
the sequence. 

- Multivariate MLP: More than 1 observation for each time step. 2 models:

    - Multiple Input Series: Have two or more parallel input time series and an output time series that is dependent on the input time series.
    [a1, a2, ..., an], [b1, b2, ..., bn] -> output series: [c1, c2, ..., cn]

    Example:

    Input:
        10, 15   
        20, 25   
        30, 35    
    Output: 65

    In general:

        [[10 15]
        [20 25]
        [30 35]] 65
        [[20 25]
        [30 35]
        [40 45]] 85
        [[30 35]
        [40 45]
        [50 55]] 105
        [[40 45]
        [50 55]
        [60 65]] 125
        [[50 55]
        [60 65]
        [70 75]] 145
        [[60 65]
        [70 75]
        [80 85]] 165
        [[70 75]
        [80 85]
        [90 95]] 185

    - (*) Multiple Parallel Series / Multivariate forecasting:
    Case where there are multiple parallel time series and a value must be predicted for each. Example: [(a1,b1), (a2,b2), (a3,b3),..., (an, bn)] and we want to predict
    --> (a_n+1, b_n+1) - the next value for each of the series (a and b series).  
    HOW? -> 2 approaches:
        - Predict with 1 single MLP.
        - Predict each output series with a different MLP (1 per series).


- Multi-Step MLP:

Note: *In practice, there is little difference to the MLP model in predicting a vector output that represents different output variables (as in the previous example) or a vector output that represents multiple time steps of one variable. Nevertheless, there are subtle and important differences in the way the training data is prepared.*

Both the input and output components will be comprised of multiple time steps
(and may or may not have the same number of steps).

Having [x1, x2, ..., xn] it predicts the next num_out time step values (x_n+1, ..., x_n+num_out)

Example: Given [10, 20, 30, 40, 50, 60, 70, 80, 90] we could used the last 3 time
steps as input and forecast the next 2 time steps. 

Input: [10, 20, 30]

Output: [40, 50]

Therefore the idea would be to have a model that given a 3 time steps could predict
the next 2 time steps: [70,80,90] --> output --> (100,110).

- Multivariate Multi-Step MLP:

Variants:

    - Multiple Input Multi-Step Output:
    Multivariate time series forecasting problems where the output series is separate but dependent upon the input time series, and multiple time steps are required for the output series.

    Example:
    [[ 10  15  25]
    [ 20  25  45]
    [ 30  35  65]
    [ 40  45  85]
    [ 50  55 105]
    [ 60  65 125]
    [ 70  75 145]
    [ 80  85 165]
    [ 90  95 185]]

    Input:     10, 15
               20, 25
               30, 35

    Output:    65
               85

    - Multiple Parallel Input and Multi-Step Output:

    Given multiple parallel input series:
    [(a1, b1, c1), (a2, b2, c2), (a3, b3, c3)...(an, bn, cn)] 
    predict as output the next num_out time steps of each of the input 
    series: [(a_n+1, b_n+1, c_n+1), (a_n+2, b_n+2, c_n+2), ..., (a_n+num_out, b_n+num_out, c_n+num_out)]

    Input:
        10, 15, 25
        20, 25, 45
        30, 35, 65

    Output: 
        40, 45, 85
        50, 55, 105

## Initial simple example - Univariate MLP model

Given a time series: [x1, x2, ..., xn] it predicts the next output value (x_n+1)

Example: Input data: [2 4 6 8 6 4 3 3 5 7]

1. Split the data in windows - to provide it so as training data for training the MLP

e.g. window size = 3: [2,4,6][8], [4,6,8][6], [6,8,6][4]...

...

## Our model

Okay, so, from all the mentioned models about what are we going to do?

We suggest on doing a model based on the Multiple Parallel Series / Multivariate forecasting,
that is the one that having as input:

`[(a1,b1), (a2,b2), (a3,b3),..., (an, bn)]` and we want to predict --> (a_n+1, b_n+1) - the next value for each of the series (a and b series).

And, among the 2 approaches:
- (1) Predict with 1 single MLP. ~*Assuming a and b are dependent*~
- (2) Predict each output series with a different MLP (1 per series). ~*Assuming a and b are independent*~

### 1. 1 single MLP model predictor

We are going to take the first one, since we understand that this way the model
learns that both time series `a` and `b` are relationed, and that the outputs,
although different depend on a joint input [(a1,b1), (a2,b2), ...,(aw,bw)].

TODO: check if this really is the case when using a single MLP for both series output.

#### Model

- **Window-based**: Training with the window division of the series approach: that is, 
the learning is based on the idea of providing a windowed slice of the series as 
input (e.g. w=3, input:[(a1,b1), (a2,b2), (a3,b3)] we predict ŷ = (a4,b4) to
match against y = (a4,b4)).

    - TODO: Test different window sizes. 

- **Testing**: Building of the test predicted output set ŷ_test, to match against
the test output set. *How?*: Assuming we do not have the real test data to do
the predictions to obtain all the test output values.

Example: *for the lynx and hare dataset*

Assume we have:

- |Train| = 45, |Test| = 12
- window size w = 3

So, first we build the model (weights and bias fit process of the MLP).
Then we will try to check the model against the test data (not used for the 
training process).
For that, we will predict all the test values in the following manner:

(we need to predict: x_46 until x_57)

- [x_43, x_44, x_45] -> x_46'
- [x_44, x_45, x_46'] -> x_47'
- [x_45, x_46', x_47'] -> x_48'
- [x_46', x_47', x_48'] -> x_49'
- ...
- [x_54', x_55', x_56'] -> x_57'

**Note that:** we are not using the real test data values to get the prediction
of a test output value, but the self-predicted values.

Once we have this predicted |test| output set, we compare it against the real
|test| dataset to obtain the error and measure the performance.



### Improving the model - Tuning parameters

| #epochs | Optimizer (learning rate) | Window size | hiddenNeurons |                                  MSE                                 |
|:-------:|:-------------------------:|:-----------:|:-------------:|:--------------------------------------------------------------------:|
|   1000  |   Gradient_Descent(0.01)  |      3      |       15      |  1345 + 266 = 1611  (BAD - GD gives  a straight line as prediction!) |
|   1000  |         ADAM(0.01)        |      3      |       15      |                           880 + 176 = 1056                           |
|   100   |         ADAM(0.01)        |      3      |       15      |                          1150 + 589 = 1739.8                         |
|   500   |         ADAM(0.01)        |      3      |       15      |                           781 + 284 = 1066                           |
|   500   |         ADAM(0.01)        |      10     |      150      |    9729 + 1941 = 11671  (overfitted - error ~ 0 for train dataset)   |
|   500   |         ADAM(0.01)        |      10     |       50      |  8335 + 5048 = 13383  (overfitted - error ~ 10^-2 for train dataset) |
|   500   |         ADAM(0.01)        |      10     |       50      |  8335 + 5048 = 13383  (overfitted - error ~ 10^-2 for train dataset) |
|   500   |         ADAM(0.01)        |      5      |       50      | 18433 + 2724 = 21158  (overfitted - error ~ 10^-2 for train dataset) |
|   500   |         ADAM(0.01)        |      5      |       15      |    5455 + 753 = 6208  (overfitted - error ~ 10¹ for train dataset)   |
|   100   |         ADAM(0.01)        |      5      |       15      |                           809 + 537 = 1347                           |
|   200   |         ADAM(0.01)        |      3      |       15      |                          1116 + 373 = 1116                           |
|   200   |         ADAM(0.01)        |      3      |       5       |                 1225 (quite flat shape of the curve)                 |
|   200   |         ADAM(0.01)        |      3      |       25      |                                 1244                                 |
|   500   |         ADAM(0.01)        |      3      |       25      |                                 888                                  |










