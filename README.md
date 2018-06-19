# Hyperparameter optimization in Deep Neural Networks
## In the context of EEG classification

This is the repository for my Bachelor's Thesis, completed on June 18th 2018.

### Abstract (English)

With the increase in computational power in the last years, neural networks have seen a grand revival that has brought along significant progress in many problems that were once considered too hard. However, this newfound power comes hand in hand with a raise in the difficulty of finding the optimal configuration for a given task.

In this work we propose an incremental evolutionary method for optimizing hyperparameters in neural networks. Starting with a feature selection phase, a population of neural networks is first evolved to find appropriate hidden layer configurations (*structure optimization*); then, another population is evolved in order to find the best combination of learning rate, dropout rate and number of epochs (*learning optimization*). Every one of these three steps can leverage implicit GPU parallelism and explicit CPU task distribution if the computational loads demand it.

Applied to classification of motor imagery tasks for brain-computer interfacing (*BCI*), this method appears to produce very promising results in terms of accuracy, both using neural networks and some simpler classifiers like Support Vector Machines.

__Keywords__: Brain-computer interfaces (BCI) · Genetic algorithms · Artificial neural networks · Feature selection · Hyperparameter optimization


### Abstract (Spanish)

Gracias a los avances de los últimos años en potencia de cómputo, el campo de las redes neuronales artificiales ha tenido un importante resurgimiento que ha traído grandes progresos en muchos problemas que se creían demasiado difíciles. Sin embargo, esta potencia renovada trae consigo un incremento en la dificultad de encontrar la configuración más apropiada para cada problema en concreto.

En este trabajo se propone un método evolutivo incremental para optimizar hiperparámetros en redes neuronales. Tras una fase previa de selección de características, se hace evolucionar una primera población de redes neuronales para encontrar arquitecturas que proporcionen un rendimiento mayor al inicial. Después, en una segunda fase se hace evolucionar otra población con el objetivo de encontrar la mejor combinación de ratio de aprendizaje, tasa de *dropout* y épocas de entrenamiento. Adicionalmente, en cada una de estas fases es posible aprovechar o bien el paralelismo implícito de una GPU o bien el paralelismo a nivel de distribución de tareas a distintas CPUs.

En su aplicación a clasificación en visualización motora usando interfaces cerebro-máquina, este método parece conseguir resultados muy prometedores en cuanto a precisión, usando tanto redes neuronales como clasificadores más simples como SVM.

__Palabras clave__: Interfaces cerebro-máquina (BCI) · Algoritmos genéticos · Redes neuronales artificiales · Selección de características · Optimización de hiperparámetros

### Contents

+ Folder *src* contains all __Python__ source code written throughout the duration of this project.
+ In folder *doc* you can find the proper __Thesis__ document and all __R__ scripts used to create charts.
+ Finally, all the experimental results are stored in folder *results*.

### Technical acknowledgments

This project has been partially funded by grant TIN2015-67020-P (Spanish *Ministerio de Economía y Competitividad* and *European Regional Development Fund*).
