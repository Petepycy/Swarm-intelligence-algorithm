import numpy as np
import random

# Read TSP instance data from a file
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
            city_number = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            city_coordinates.append((x, y))

    return dimension, city_coordinates

# ACO algorithm to solve TSP
def aco_tsp(dimension, city_coordinates, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, rho=0.1, Q=100):
    # Initialization
    num_cities = dimension
    pheromone_matrix = np.ones((num_cities, num_cities)) / 1000  # Initial pheromone levels
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
                # Debugging code
                print(f"num_cities: {num_cities}")
                print(f"len(tour): {len(tour)}")
                prob = np.zeros(num_cities - len(tour) + 1)
                print(f"Size of prob: {len(prob)}")

                for city in unvisited_cities:
                    prob[city] = (pheromone_matrix[current_city, city] ** alpha) * ((1.0 / distance(city_coordinates[current_city], city_coordinates[city])) ** beta)

                prob /= prob.sum()
                next_city = np.random.choice(list(unvisited_cities), p=prob)
                tour.append(next_city)
                unvisited_cities.remove(next_city)
                current_city = next_city

            ant_tours.append(tour)

        # Update pheromone levels
        pheromone_matrix *= (1 - rho)
        for tour in ant_tours:
            tour_length = sum(distance(city_coordinates[tour[i]], city_coordinates[tour[i + 1]]) for i in range(num_cities - 1))
            tour_length += distance(city_coordinates[tour[-1]], city_coordinates[tour[0]])
            if tour_length < best_tour_length:
                best_tour = tour
                best_tour_length = tour_length
            for i in range(num_cities - 1):
                pheromone_matrix[tour[i], tour[i + 1]] += (Q / tour_length)
            pheromone_matrix[tour[-1], tour[0]] += (Q / tour_length)

    return best_tour, best_tour_length

# Calculate the Euclidean distance between two cities
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

# List of TSP instances to solve
tsp_instances = ["xqf131.tsp", "xqg237.tsp", "pma343.tsp", "pka379.tsp", "bcl380.tsp", "pbl395.tsp"]

for instance in tsp_instances:
    dimension, city_coordinates = read_tsp_instance(instance)

    best_tour, best_tour_length = aco_tsp(dimension, city_coordinates)

    # Print the best tour and its length
    print(f"{instance}: Best tour: {best_tour}, Best tour length: {best_tour_length}")
