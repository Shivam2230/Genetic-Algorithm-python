import numpy
import numpy as np

#Aim: To maximize equation y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
#Inputs:(x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
#calculate weights w1 to w6 using genetic algo

inp=[4,-2,3.5,5,-11,-4.7]

num_weights=len(inp)
sol_per_pop=8
parents_size=4
best_op=[]

pop_size=(sol_per_pop,num_weights)
new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)
print(new_population)

def fitness_func(inp,pop):
    return np.sum(inp*pop, axis=1)

def select_mating_pool(pop, fitness, num_parents):
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover


for i in range(100):
    print(new_population)
    print("Generation : ",i)
    fitness=fitness_func(inp,new_population)
    print("Fitness")
    print(fitness)
    print("Best Result : ")
    print(max(fitness))
    best_op.append(max(fitness))
    parents=select_mating_pool(new_population, fitness,parents_size)
    print("Parents")
    print(parents)
    offspring_crossover=crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    print("Crossover")
    print(offspring_crossover)
    offspring_mutation = mutation(offspring_crossover, num_mutations=2)
    print("Mutation")
    print(offspring_mutation)
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
fitness = fitness_func(inp, new_population)
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])


import matplotlib.pyplot
matplotlib.pyplot.plot(best_op)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()

