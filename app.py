import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from stable_baselines3 import PPO

#Load Datasets
orders_df = pd.read_csv('./orders.csv')
shelves_df = pd.read_csv('./Shelves.csv')
robots_df = pd.read_csv('./robots.csv')

shelves_locations = shelves_df[['Location_X', 'Location_Y']].to_numpy()
num_shelves = len(shelves_locations)

def shelf_distance(s1, s2):
    return np.linalg.norm(shelves_locations[s1] - shelves_locations[s2])

#Distance Matrix
distance_matrix = np.zeros((num_shelves, num_shelves))
for i in range(num_shelves):
    for j in range(num_shelves):
        distance_matrix[i][j] = shelf_distance(i, j)

#Genetic Algorithm for Order Allocation
class GeneticOrderAllocator:
    def __init__(self, orders_df, robots_df, shelves_df, generations=50, population_size=20):
        self.orders_df = orders_df
        self.robots_df = robots_df
        self.shelves_df = shelves_df
        self.generations = generations
        self.population_size = population_size
        self.num_robots = len(robots_df)

    def fitness(self, allocation):
        load_balance = max([allocation.count(r) for r in range(self.num_robots)]) - min([allocation.count(r) for r in range(self.num_robots)])
        total_distance = 0
        for idx, robot_id in enumerate(allocation):
            order = self.orders_df.iloc[idx]
            shelf_id = random.choice(range(len(self.shelves_df)))
            robot_pos = shelves_locations[robot_id % len(shelves_locations)]
            shelf_pos = shelves_locations[shelf_id]
            total_distance += np.linalg.norm(robot_pos - shelf_pos)
        return -(load_balance + 0.01 * total_distance)

    def run(self):
        population = [random.choices(range(self.num_robots), k=len(self.orders_df)) for _ in range(self.population_size)]
        for _ in range(self.generations):
            fitness_scores = [self.fitness(p) for p in population]
            selected = [population[i] for i in np.argsort(fitness_scores)[-self.population_size//2:]]
            children = []
            for _ in range(self.population_size//2):
                parent1, parent2 = random.sample(selected, 2)
                cross_point = random.randint(0, len(parent1) - 1)
                child = parent1[:cross_point] + parent2[cross_point:]
                children.append(child)
            population = selected + children
        best_allocation = population[np.argmax([self.fitness(p) for p in population])]
        return best_allocation

# Apply GA for order allocation
ga_allocator = GeneticOrderAllocator(orders_df, robots_df, shelves_df)
best_allocation = ga_allocator.run()

orders_to_robots = {i: [] for i in range(len(robots_df))}
for idx, robot_id in enumerate(best_allocation):
    orders_to_robots[robot_id].append(orders_df.iloc[idx])

# Display Order Allocation Summary
print("\n=== Optimized Order Allocation ===")
allocation_data = []
for robot_id, orders in orders_to_robots.items():
    for order in orders:
        allocation_data.append([robot_id, order['Order_ID'], order['Number_of_Items']])
df_alloc = pd.DataFrame(allocation_data, columns=['Robot_ID', 'Order_ID', 'Number_of_Items'])
print(df_alloc.sort_values(by='Robot_ID'))

#Ant Colony Optimization (Shelf Selection)
class AntColony:
    def __init__(self, distance_matrix, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.distances = distance_matrix
        self.pheromone = np.ones(self.distances.shape) / len(distance_matrix)
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        all_time_shortest_path = ([], np.inf)
        for _ in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone *= self.decay
        return all_time_shortest_path

    def spread_pheromone(self, all_paths):
        for path, dist in sorted(all_paths, key=lambda x: x[1])[:self.n_best]:
            for i in range(len(path) - 1):
                move = (path[i], path[i+1])
                if self.distances[move[0]][move[1]] > 0:
                    self.pheromone[move[0]][move[1]] += 1.0 / self.distances[move[0]][move[1]]

    def gen_path_dist(self, path):
        return sum(self.distances[path[i]][path[i+1]] for i in range(len(path) - 1))

    def gen_all_paths(self):
        return [(self.gen_path(0), self.gen_path_dist(self.gen_path(0))) for _ in range(self.n_ants)]

    def gen_path(self, start):
        path, visited, prev = [start], set([start]), start
        for _ in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append(move)
            visited.add(move)
            prev = move
        path.append(start)
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        dist = np.where(dist == 0, np.inf, dist)
        row = (pheromone ** self.alpha) * ((1.0 / dist) ** self.beta)
        if np.sum(row) == 0:
            return random.choice(list(set(range(len(self.distances))) - visited))
        return np.random.choice(range(len(self.distances)), p=row / row.sum())

aco = AntColony(distance_matrix, n_ants=10, n_best=3, n_iterations=50, decay=0.95)
best_shelf_route = aco.run()

# Display Shelf Route
print("\n=== Optimized Shelf Picking Route ===")
route, total_dist = best_shelf_route
print("Shelf Visit Order:", ' -> '.join(map(str, route)))
print(f"Total Distance: {total_dist:.2f} units")

# Visualize Shelf Route Distances
plt.figure(figsize=(8, 4))
colors = plt.cm.viridis(np.linspace(0, 1, len(route) - 1))
distances = [shelf_distance(route[i], route[i+1]) for i in range(len(route) - 1)]
plt.bar(range(len(distances)), distances, color=colors)
plt.title('Distance Between Shelf Picks')
plt.xlabel('Step')
plt.ylabel('Distance')
plt.show()

# Single-Agent RL Training per Robot
class WarehouseEnv(gym.Env):
    def __init__(self, robot_id, assigned_orders):
        super(WarehouseEnv, self).__init__()
        self.robot_id = robot_id
        self.assigned_orders = assigned_orders
        self.action_space = gym.spaces.Discrete(num_shelves)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)

    def reset(self):
        self.state = np.zeros(2)
        return self.state

    def step(self, action):
        distance_to_shelf = shelf_distance(int(self.state[0]), action)
        reward = -distance_to_shelf
        self.state[0] = action
        done = False
        return self.state, reward, done, {}

# Train PPO for each robot individually
models = {}
for robot_id, assigned_orders in orders_to_robots.items():
    env = WarehouseEnv(robot_id, assigned_orders)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    models[robot_id] = model
    print(f"[✔] Robot {robot_id}: PPO Training Completed (Orders Assigned: {len(assigned_orders)})")


#Visualize Robot Order Load
robot_ids = list(orders_to_robots.keys())
order_counts = [len(orders_to_robots[robot]) for robot in robot_ids]

plt.figure(figsize=(8, 4))
plt.bar(robot_ids, order_counts, color='skyblue')
plt.title('Orders Assigned per Robot')
plt.xlabel('Robot ID')
plt.ylabel('Number of Orders')
plt.xticks(robot_ids)
plt.grid(axis='y')
plt.show()