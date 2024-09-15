# implement_genetic_algorithm
# Create a genetic algorithm to solve a specific optimization problem, such as the traveling salesman problem. Include selection, crossover, mutation, and fitness evaluation functions.

import random
import numpy as np

# Define the distance matrix (example with 5 cities)
distance_matrix = [
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 4],
    [10, 4, 8, 0, 5],
    [7, 3, 4, 5, 0]
]

# Number of cities
num_cities = len(distance_matrix)

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.05

# Generate a random route (chromosome)


def generate_chromosome():
    chromosome = list(range(num_cities))
    random.shuffle(chromosome)
    return chromosome

# Calculate the total distance of a route (fitness function)


def calculate_distance(route):
    total_distance = 0
    for i in range(len(route)):
        total_distance += distance_matrix[route[i]
                                          ][route[(i + 1) % num_cities]]
    return total_distance

# Generate initial population


def generate_initial_population():
    return [generate_chromosome() for _ in range(POPULATION_SIZE)]

# Selection: Select parents using tournament selection


def selection(population):
    selected = []
    for _ in range(POPULATION_SIZE):
        # Tournament of 5
        tournament = random.sample(population, 5)
        # Select the one with the best (shortest) distance
        winner = min(tournament, key=calculate_distance)
        selected.append(winner)
    return selected

# Crossover: Order Crossover (OX)


def crossover(parent1, parent2):
    size = len(parent1)
    # Randomly select two crossover points
    start, end = sorted(random.sample(range(size), 2))

    # Create a child with a sub-route from parent1
    child = [-1] * size
    child[start:end + 1] = parent1[start:end + 1]

    # Fill in the rest from parent2, maintaining the order
    current_pos = (end + 1) % size
    for city in parent2:
        if city not in child:
            child[current_pos] = city
            current_pos = (current_pos + 1) % size

    return child

# Mutation: Swap Mutation


def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            # Swap two random cities
            j = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

# Main Genetic Algorithm


def genetic_algorithm():
    # Step 1: Generate initial population
    population = generate_initial_population()

    # Step 2: Run for a set number of generations
    for generation in range(GENERATIONS):
        # Step 3: Evaluate the population
        population.sort(key=calculate_distance)

        # Print the best route and its distance every 50 generations
        if generation % 50 == 0:
            best_route = population[0]
            best_distance = calculate_distance(best_route)
            print(f"Generation {generation}: Best Distance = {best_distance}")

        # Step 4: Selection
        selected_population = selection(population)

        # Step 5: Crossover
        next_generation = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]

            # Perform crossover to generate two children
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)

            # Step 6: Mutation
            mutate(child1)
            mutate(child2)

            # Add children to the next generation
            next_generation.append(child1)
            next_generation.append(child2)

        # Update population for the next generation
        population = next_generation

    # Final result
    best_route = population[0]
    best_distance = calculate_distance(best_route)
    print(f"Final Best Route: {best_route}")
    print(f"Final Best Distance: {best_distance}")


if __name__ == "__main__":
    genetic_algorithm()
