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

1. main.py - to start training snake game using genetic algorithm
2. snake_game.py - contains logic of creating snake game using pygame
3. run_game.py - play snake game using predicted directions from genetic algorithm
4. ga.py - contains genetic algorithm functions like crossover, mutation etc
5. ffnn.py - contains the functions for calculating the output from feed forward neural network

**Explanation:**

**Creating a snake game and deciding neural network architecture**
Snake game is created with pygame and network architecture with 7 units in the input layer, 3 units in the output layer with ‘softmax’ and used 2 hidden layers one of 9 units and other of 15 units with ‘relu’ as shown below.

![alt text](https://i0.wp.com/theailearner.com/wp-content/uploads/2018/11/Snake_game_with_neural_network_with_3_outputs.png?w=763&ssl=1)

**Creating Initial Population**

Here, I have chosen 50 individuals in the population and each individual is an array of weights of the neural network. Randomly initialize these individuals. Code,

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

**Deciding the Fitness Function**

Any fitness function can be used here, I've used the following fitness function:
“On every grasp of food, I have given 5000 reward points and if it collides with the boundary or itself, I have awarded a penalty of 150 points“

**Evaluating the population**

For each individual, a game is played and the fitness function is calculated which is then appended in the list as shown in the code below,

```python
def cal_pop_fitness(pop):
    fitness = []
    for i in range(pop.shape[0]):
        fit = run_game_with_ML(display,clock,pop[i])
        print('fitness value of chromosome '+ str(i) +' :  ', fit)
        fitness.append(fit)
    return np.array(fitness)
 
fitness = cal_pop_fitness(new_population)
```
**Selection, Crossover, and Mutation**

**Selection:** Now, according to fitness value, some best individuals will be selected from the population and are stored in the ‘parents’ array as shown in the code below,

```python
def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999
    return parents

parents = select_mating_pool(new_population, fitness, num_parents_mating)
```

**Crossover:** To produce children for the next generation, the crossover is used. First, two individuals are randomly selected from the best, then I randomly choose some values from first and some from the second individual to produce new offspring. This process is repeated until the total population size is not achieved as shown in the code below,

```python
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)

    for k in range(offspring_size[0]):

        while True:
            parent1_idx = random.randint(0, parents.shape[0] - 1)
            parent2_idx = random.randint(0, parents.shape[0] - 1)
            if parent1_idx != parent2_idx:
                for j in range(offspring_size[1]):
                    if random.uniform(0, 1) &lt; 0.5:
                        offspring[k, j] = parents[parent1_idx, j]
                    else:
                        offspring[k, j] = parents[parent2_idx, j]
                break
    return offspring
    
# Generating next generation using crossover.
offspring_crossover = crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))
```

**Mutation:** Then, some variations are being added to the newly formed offspring. Here, for each child, I randomly selected 25 weights and mutated them by adding some random value as shown in the code below,

```python
def mutation(offspring_crossover):

    for idx in range(offspring_crossover.shape[0]):
        for _ in range(25):
            i = randint(0,offspring_crossover.shape[1]-1)

        random_value = np.random.choice(np.arange(-1,1,step=0.001),size=(1),replace=False)
        offspring_crossover[idx, i] = offspring_crossover[idx, i] + random_value

    return offspring_crossover
    
offspring_mutation = mutation(offspring_crossover)
```

**New Population Created**

With the help of fitness function, crossover and mutation, new population for the next generation is created. Now, next thing is to replace the previous population with this newly formed. Code,

```python
# Creating the new population based on the parents and offspring.
new_population[0:parents.shape[0], :] = parents
new_population[parents.shape[0]:, :] = offspring_mutation
```

Now we will repeat this process until our target for certain game score is not achieved. In this problem, I have used 100 generations for training and 2500 steps in a game, which was able to achieve a maximum score of 40. You can choose more number of steps per game and can achieve more score.





