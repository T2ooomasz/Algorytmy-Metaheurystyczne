if __name__ == '__main__':
    pop = population(200)
    parents = pop.initialize_population()
    phenotypes = []
    for i in range(200):
        phenotypes.append(parents[i].phenotype)
    print(min(phenotypes))   
    gen = genetic_algorithm('roulette', 'roulette', 'PMX', parents)
    new_population = gen.parents
    phenotypes1 = []
    for i in range(len(new_population)):
        phenotypes1.append(new_population[i].phenotype)
    print(min(phenotypes1))
