# Hybrid Algorithm for Traveling Salesman Problem
In a Traveling Salesman Problem (TSP), suppose that a salesman is given a set of n cities and a distance matrix, in which the matrix elements denote the distances among cities, the salesman is required to make a Hamiltonian tour with minimum distance. The Hamiltonian tour is also referred to as a round trip tour whereby the salesman must visit each city once and only once, and eventually returns to the starting city. The naïve implementation for this task is considering each combination of different cities and making sure such combination possesses the necessary properties for a Hamiltonian tour. This brute force solution has the time complexity of O(n!). In short, The TSP is an NP-Hard issue, making it impossible for many cities to discover a precise solution.

Genetic algorithm (GA) is one of the metaheuristic algorithms which mimic evolution process according to the concept of the survival of the fittest. The basic idea of implementing GA in TSP is to encode the viable solutions as chromosomes and then employ genetic operators like crossover and mutation to create potential solutions. The GA continuously assesses the solutions' fitness through several iterations of evolutionary cycles across many generations and uses the results to direct its search for better solution. In other words, the GA helps in discovering an approximation of a good solution. While GA may handle huge data sets, they are prone to slowed evolution and early convergence on a local optimum, preventing further exploration. To address these issues, we are trying the hybrid algorithm, combining GA and fuzzy logic, which makes use of hybridization of several approaches, the integration technique seeks to overcome the constraints of individual techniques.
## Dataset
National TSP datasets taken from the URL https://www.math.uwaterloo.ca/tsp/world/countries.html. For these datasets, the cost of travel between cities is specified by the Euclidean distance rounded to the nearest whole number.
## Algorithm Design
The algorithm will start from reading the input tsp dataset and store the coordinate of cities inside a list. The distance between each city is computed and create a distance matrix. A city will be chosen as starting point and the greedy method is employed for chromosome initialization in GA. Rather than selecting a city at random, the greedy algorithm develops a solution by continually selecting the next nearest city to the existing one. This produces a solution that is likely to be near to the optimal solution, which provide a good start for the algorithm especially dealing with larger number of cities.

By sum up the travel distance that identified from the distance matrix, the entire distance route can be calculated. This distance is then divided by the optimal distance (identified from the websites) to convert it into fitness value in the range from 0 to 1 (approaching to 1 means closer to the identified optimal solution). For GA, elitism is used in the algorithm to preserve a certain number of the best solutions, called "elites," from the current generation and including them in the next generation without any modification. The population is evaluated, and the fitness scores of each solution are sorted. A ratio is selected to select certain number of the best solutions to create for next generation. This technique helps the GA to converge to an optimal solution more quickly and efficiently. After this, the roulette wheel selection method is used to select solutions for the genetic operations such as crossover and mutation.

The fuzzy logic system is converting the fitness value into fuzzy input through fuzzification. In general, this fuzzy logic controller will provide the value of population number, crossover rate and mutation rate adapted from different degree of fitness value at each generation. For the inference rules, the input is contracting with the outputs. For examples, if the fitness value is low, the population size, crossover rate and mutation rate will be high. If the fitness value is medium then the population size, crossover rate and mutation rate will also be medium. By going through the inference rules and defuzzification, the output crisp value can used to estimate the size of the population that is needed to find an optimal solution. With fuzzy logic, the crossover rate can be adapted across the generation. This allows for more precise control over the amount of genetic information that is exchanged between the parents. Also with fuzzy logic, mutation can be performed by introducing changes with different probabilities, allowing for more precise control over the amount of change introduced.

After the defined iteration number is reached, the hybrid algorithm will stop and show the found shortest distance and route.

## Testing
There are some variables need to be defined before algorithm run:
1. Download the input file and state its directory 
2. Key in current known optimal route length
3. Key in number of iteration 
