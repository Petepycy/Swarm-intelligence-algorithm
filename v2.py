import numpy as np
import random

# Function to read TSP instance data from a file
def read_tsp_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    dimension = None
    city_coordinates = []

    read_coordinates = False

    for line in lines:
        line = line.strip()
        if line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1])
        elif line == "NODE_COORD_SECTION":
            read_coordinates = True
        elif read_coordinates and line != "EOF":
            parts = line.split()
            x, y = float(parts[1]), float(parts[2])
            city_coordinates.append((x, y))

    return dimension, city_coordinates

# Calculate the Euclidean distance between two cities
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

# Calculate the total length of a tour
def calculate_tour_length(tour, city_coordinates):
    length = 0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)  # Ensure it's cyclic
        length += distance(city_coordinates[tour[i]], city_coordinates[tour[j]])
    return length

# Improved ACO algorithm to solve TSP
def improved_aco_tsp(dimension, city_coordinates, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, rho=0.1, Q=100):
    # Initialization
    num_cities = dimension
    pheromone_matrix = np.ones((num_cities, num_cities)) / num_cities  # Adjusted initial pheromone levels
    best_tour = None
    best_tour_length = float('inf')

    # Main ACO loop
    for iteration in range(num_iterations):
        ant_tours = []

        # Ants construct tours
        for ant in range(num_ants):
            current_city = random.randint(0, num_cities - 1)
            tour = [current_city]
            unvisited_cities = set(range(num_cities))
            unvisited_cities.remove(current_city)

            while unvisited_cities:
                probabilities = []
                for city in unvisited_cities:
                    prob = (pheromone_matrix[current_city, city] ** alpha) * \
                           ((1.0 / distance(city_coordinates[current_city], city_coordinates[city])) ** beta)
                    probabilities.append(prob)

                probabilities = np.array(probabilities) / np.sum(probabilities)
                next_city = np.random.choice(list(unvisited_cities), p=probabilities)
                tour.append(next_city)
                unvisited_cities.remove(next_city)
                current_city = next_city

            ant_tours.append(tour)

        # Update pheromone levels
        pheromone_matrix *= (1 - rho)
        for tour in ant_tours:
            tour_length = calculate_tour_length(tour, city_coordinates)
            if tour_length < best_tour_length:
                best_tour = tour
                best_tour_length = tour_length

            for i in range(len(tour)):
                j = (i + 1) % len(tour)  # Ensure it's cyclic
                pheromone_matrix[tour[i], tour[j]] += Q / tour_length

    return best_tour, best_tour_length

# Example usage with a TSP file
file_path = 'xqf131.tsp'  # Replace with your TSP file path
dimension, city_coordinates = read_tsp_instance(file_path)
best_tour, best_tour_length = improved_aco_tsp(dimension, city_coordinates)
print(f"Best tour: {best_tour}, Best tour length: {best_tour_length}")
