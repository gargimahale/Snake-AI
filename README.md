# Snake-GA-
Where Genetic Algorithm (GA) and Neural Network(NN) play the snake game 

The advantage of using GA + NN, is that we don’t need any training data.
We'll use GA in NN, instead of backpropagation, we will update weights using GA.

**Using GA in NN:**

**Steps:**

1. Creating a snake game and deciding neural network architecture.
2. Creating an initial population.
3. Deciding the fitness function.
4. Play a game for each individual in the population and sort each individual in the population based on the fitness function score.
5. Select a few top individuals from the population and create the remaining population from these top selected individuals using Crossover and mutation.
6. The new population is created (meaning the next generation).
7. Go to step 4 and repeat until the stopping criteria are not satisfied.

**Files:**

main.py - to start training snake game using genetic algorithm
snake_game.py - contains logic of creating snake game using pygame
run_game.py - play snake game using predicted directions from genetic algorithm
ga.py - contains genetic algorithm functions like crossover, mutation etc
ffnn.py - contains the functions for calculating the output from feed forward neural network

**Explanation:**

**Creating a snake game and deciding neural network architecture**
Snake game is created with pygame and network architecture with 7 units in the input layer, 3 units in the output layer with ‘softmax’ and used 2 hidden layers one of 9 units and other of 15 units with ‘relu’ as shown below.

![alt text](https://i0.wp.com/theailearner.com/wp-content/uploads/2018/11/Snake_game_with_neural_network_with_3_outputs.png?w=763&ssl=1)

**Creating Initial Population**

Here, I have chosen 50 individuals in the population and each individual is an array of weights of the neural network. Randomly initialize these individuals. 
```python
# n_x   no. of input units
# n_h   no. of units in hidden layer 1
# n_h2  no. of units in hidden layer 2
# n_y   no. of output units

# The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
sol_per_pop = 50
num_weights = n_x*n_h + n_h*n_h2 + n_h2*n_y

# Defining the population size.
pop_size = (sol_per_pop,num_weights)
#Creating the initial population.
new_population = np.random.choice(np.arange(-1,1,step=0.01),size=pop_size,replace=True)
```








