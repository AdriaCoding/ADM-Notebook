

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