import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn")

np.random.seed(42)

cities = [0, 1, 2, 3, 4]

adjacency_mat = np.asarray(
    [
        [0.00, 28.02, 17.12, 27.46, 46.07],
        [28.02, 0.00, 34.00, 25.55, 25.55],
        [17.12, 34.00, 0.00, 18.03, 57.38],
        [27.46, 25.55, 18.03, 0.00, 51.11],
        [46.07, 25.55, 57.38, 51.11, 0.00],
    ]
)

class Population():
    def __init__(self, bag, adjacency_mat):
        self.bag = bag
        self.parents = []
        self.score = 0
        self.best = None
        self.adjacency_mat = adjacency_mat

def init_population(cities, adjacency_mat, n_population):
    return Population(
        np.asarray([np.random.permutation(cities) for _ in range(n_population)]),
        adjacency_mat
    )

pop = init_population(cities, adjacency_mat, 5)
pop.bag

def fitness(self, chromosome):
    return sum(
        [
            self.adjacency_mat[chromosome[i], chromosome[i + 1]]
            for i in range(len(chromosome) - 1)
        ]
    )

Population.fitness = fitness

def evaluate(self):
    distances = np.asarray(
        [self.fitness(chromosome) for chromosome in self.bag]
    )
    self.score = np.min(distances)
    self.best = self.bag[distances.tolist().index(self.score)]
    self.parents.append(self.best)
    if False in (distances[0] == distances):
        distances = np.max(distances) - distances
    return distances / np.sum(distances)


Population.evaluate = evaluate
pop.evaluate()
pop.best
pop.score

def select(self, k=4):
    fit = self.evaluate()
    while len(self.parents) < k:
        idx = np.random.randint(0, len(fit))
        if fit[idx] > np.random.rand():
            self.parents.append(self.bag[idx])
    self.parents = np.asarray(self.parents)

Population.select = select
pop.select()
pop.parents

def swap(chromosome):
    a, b = np.random.choice(len(chromosome), 2)
    chromosome[a], chromosome[b] = (
        chromosome[b],
        chromosome[a],
    )
    return chromosome

def crossover(self, p_cross=0.1):
    children = []
    count, size = self.parents.shape
    for _ in range(len(self.bag)):
        if np.random.rand() > p_cross:
            children.append(
                list(self.parents[np.random.randint(count, size=1)[0]])
            )
        else:
            parent1, parent2 = self.parents[
                np.random.randint(count, size=2), :
            ]
            idx = np.random.choice(range(size), size=2, replace=False)
            start, end = min(idx), max(idx)
            child = [None] * size
            for i in range(start, end + 1, 1):
                child[i] = parent1[i]
            pointer = 0
            for i in range(size):
                if child[i] is None:
                    while parent2[pointer] in child:
                        pointer += 1
                    child[i] = parent2[pointer]
            children.append(child)
    return children

Population.crossover = crossover

def mutate(self, p_cross=0.1, p_mut=0.1):
    next_bag = []
    children = self.crossover(p_cross)
    for child in children:
        if np.random.rand() < p_mut:
            next_bag.append(swap(child))
        else:
            next_bag.append(child)
    return next_bag

Population.mutate = mutate
pop.mutate()

def genetic_algorithm(
    cities,
    adjacency_mat,
    n_population=5,
    n_iter=20,
    selectivity=0.15,
    p_cross=0.5,
    p_mut=0.1,
    print_interval=100,
    return_history=False,
    verbose=False,
):
    pop = init_population(cities, adjacency_mat, n_population)
    best = pop.best
    score = float("inf")
    history = []
    for i in range(n_iter):
        pop.select(n_population * selectivity)
        history.append(pop.score)
        if verbose:
            print(f"Generation {i}: {pop.score}")
        elif i % print_interval == 0:
            print(f"Generation {i}: {pop.score}")
        if pop.score < score:
            best = pop.best
            score = pop.score
        children = pop.mutate(p_cross, p_mut)
        pop = Population(children, pop.adjacency_mat)
    if return_history:
        return best, history
    return best

genetic_algorithm(cities, adjacency_mat, verbose=True)

best, history = genetic_algorithm(
    cities,
    adjacency_mat,
    n_iter=100,
    verbose=False,
    print_interval=20,
    return_history=True,
)

plt.plot(range(len(history)), history, color="skyblue")
plt.show()
print(best)

def generate_cities(n_cities, factor=10):
    return np.random.rand(n_cities, 2) * n_cities * factor


def make_mat(coordinates):
    res = [
        [get_distance(city1, city2) for city2 in coordinates]
        for city1 in coordinates
    ]
    return np.asarray(res)

def get_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

test_coords = [[0, 0], [0, 1], [1, 1], [1, 0]]
make_mat(test_coords)
generate_cities(5)
cities = range(100)
city_coordinates = generate_cities(len(cities))
adjacency_mat = make_mat(city_coordinates)

best, history = genetic_algorithm(
    cities, adjacency_mat, n_population=20, n_iter=1000, verbose=False, return_history=True
)

plt.plot(range(len(history)), history, color="skyblue")
plt.show()
print(best)



def print_path(best, city_coordinates):
    points = city_coordinates[best]
    x, y = zip(*points)
    plt.plot(x, y, color="skyblue", marker="o")

print_path(best, city_coordinates)

def better_generate_cities(n_cities, factor=0.2):
    x = np.asarray(range(int(-n_cities / 2), int(n_cities / 2) + 1, 1))
    y = np.sqrt(n_cities ** 2 / 4 - x ** 2)
    return np.asarray(list(zip(x, y)))

# #cities = range(100)
# city_coordinates = better_generate_cities(len(cities))
# adjacency_mat = make_mat(city_coordinates)
# best, history = genetic_algorithm(
#     cities, adjacency_mat, n_population=500, selectivity=0.05,
#     p_mut=0.05, p_cross=0.7, n_iter=100, print_interval=1, verbose=False, return_history=True
# )
# plt.plot(range(len(history)), history, color="skyblue")
# plt.show()
# print(best)
#
# print_path(best, city_coordinates)
#
# print_path(sorted(best), city_coordinates)

#----------------------------------------------------
# Open input file

infile = open('berlin52.tsp', 'r')

# Read instance header
Name = infile.readline().strip().split()[1] # NAME
FileType = infile.readline().strip().split()[1] # TYPE
Comment = infile.readline().strip().split()[1] # COMMENT
Dimension = infile.readline().strip().split()[1] # DIMENSION
EdgeWeightType = infile.readline().strip().split()[1] # EDGE_WEIGHT_TYPE
infile.readline()

# Read node list
nodelist = []
cities = int(Dimension)
for i in range(0, int(Dimension)):
    x,y = infile.readline().strip().split()[1:]
    nodelist.append([float(x), float(y)])

city_coordinates = better_generate_cities(cities)

adjacency_mat = make_mat(city_coordinates)

best, history = genetic_algorithm(
    cities, adjacency_mat, n_population=500, selectivity=0.05,
    p_mut=0.05, p_cross=0.7, n_iter=100, print_interval=1, verbose=False, return_history=True
)

plt.plot(range(len(history)), history, color="skyblue")
plt.show()
print(best)

# Close input file
infile.close()