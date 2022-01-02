import matplotlib.pyplot as plt
import TravellerSalesmanProblemandGeneticAlgorithms2 as mal

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

city_coordinates = mal.better_generate_cities(cities)

adjacency_mat = mal.make_mat(city_coordinates)
best, history = mal.genetic_algorithm(
    cities, adjacency_mat, n_population=500, selectivity=0.05,
    p_mut=0.05, p_cross=0.7, n_iter=100, print_interval=1, verbose=False, return_history=True
)





plt.plot(range(len(history)), history, color="skyblue")
plt.show()
print(best)

# Close input file
infile.close()