import random
import math
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pickle


class GA(object):
    def __init__(self, num_city, num_total, iteration, data):
        self.num_city = num_city
        self.num_total = num_total
        self.scores = []
        self.iteration = iteration
        self.location = data
        self.ga_choose_ratio = 0.5
        self.mutate_ratio = 0.6

        self.optimal_length = 0.0

        #enter current optimal result of the target city [can obtained from the website] 
        self.optimal_length = 300899.0
        
        self.dis_mat = self.compute_dis_mat(num_city, data)
        self.offsprings = self.greedy_init(self.dis_mat, num_total, num_city)
        scores = self.compute_adp(self.offsprings)
        sort_index = np.argsort(-scores)
        init_best = self.offsprings[sort_index[0]]
        init_best = self.location[init_best]
        self.iter_x = [0]
        self.iter_y = [self.optimal_length / scores[sort_index[0]]]

        # Define the membership functions for the fuzzy variables (input)
        fitness = ctrl.Antecedent(np.arange(0, 1.1, 0.001), 'fitness')
        fitness['very_low'] = fuzz.trimf(fitness.universe, [0, 0.125, 0.25])
        fitness['low'] = fuzz.trimf(fitness.universe, [0.2, 0.325, 0.45])
        fitness['medium'] = fuzz.trimf(fitness.universe, [0.4, 0.525, 0.65])
        fitness['high'] = fuzz.trimf(fitness.universe, [0.6, 0.725, 0.85])
        fitness['very_high'] = fuzz.trimf(fitness.universe, [0.8, 0.925, 1.0])

        # Define the membership functions for the mutate rate (output)
        mutate_rate = ctrl.Consequent(np.arange(0, 1.1, 0.001), 'mutate_rate', 'centroid')
        mutate_rate['very_low'] = fuzz.trimf(mutate_rate.universe, [0, 0.125, 0.25])
        mutate_rate['low'] = fuzz.trimf(mutate_rate.universe, [0.2, 0.325, 0.45])
        mutate_rate['medium'] = fuzz.trimf(mutate_rate.universe, [0.4, 0.525, 0.65])
        mutate_rate['high'] = fuzz.trimf(mutate_rate.universe, [0.6, 0.725, 0.85])
        mutate_rate['very_high'] = fuzz.trimf(mutate_rate.universe, [0.8, 0.925, 1.0])

        # Define the membership functions for the crossover rate (output)
        crossover_rate = ctrl.Consequent(np.arange(0, 1.1, 0.001), 'crossover_rate', 'centroid')
        crossover_rate['very_low'] = fuzz.trimf(crossover_rate.universe, [0, 0.125, 0.25])
        crossover_rate['low'] = fuzz.trimf(crossover_rate.universe, [0.2, 0.325, 0.45])
        crossover_rate['medium'] = fuzz.trimf(crossover_rate.universe, [0.4, 0.525, 0.65])
        crossover_rate['high'] = fuzz.trimf(crossover_rate.universe, [0.6, 0.725, 0.85])
        crossover_rate['very_high'] = fuzz.trimf(crossover_rate.universe, [0.8, 0.925, 1.0])

        # Define the membership function for the number of population (output)
        num_population = ctrl.Consequent(np.arange(10, 41, 1), 'num_population', 'centroid')
        num_population['very_low'] = fuzz.trimf(num_population.universe, [10, 15, 20])
        num_population['low'] = fuzz.trimf(num_population.universe, [15, 20, 25])
        num_population['medium'] = fuzz.trimf(num_population.universe, [20, 25, 30])
        num_population['high'] = fuzz.trimf(num_population.universe, [25, 30, 35])
        num_population['very_high'] = fuzz.trimf(num_population.universe, [30, 35, 40])

        # Define the fuzzy rules
        rule1 = ctrl.Rule(fitness['very_low'],
                          (mutate_rate['very_high'], num_population['very_high'], crossover_rate['very_high']))
        rule2 = ctrl.Rule(fitness['low'], (mutate_rate['high'], num_population['high'], crossover_rate['high']))
        rule3 = ctrl.Rule(fitness['medium'],
                          (mutate_rate['medium'], num_population['medium'], crossover_rate['medium']))
        rule4 = ctrl.Rule(fitness['high'], (mutate_rate['low'], num_population['low'], crossover_rate['low']))
        rule5 = ctrl.Rule(fitness['very_high'],
                          (mutate_rate['very_low'], num_population['very_low'], crossover_rate['very_low']))

        # Create the control system
        rule_base = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        self.fuzzy_system = ctrl.ControlSystemSimulation(rule_base)

    # random initialization
    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    # initialization using greedy method
    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            current = start_index
            
            rest.remove(current)
            result_one = [current]
            
            while len(rest) != 0:
                if len(result_one) < (int(0.95 * num_city)):
                    tmp_min = math.inf
                    tmp_choose = -1
                    for x in rest:
                        if dis_mat[current][x] < tmp_min:
                            tmp_min = dis_mat[current][x]
                            tmp_choose = x
                    current = tmp_choose
                    result_one.append(tmp_choose)
                    rest.remove(tmp_choose)
                else:
                    result_one.extend(np.random.choice(rest, num_city - len(result_one), replace=False))
                    break
            
            result.append(result_one)
            start_index += 100
        return result

    # compute the distance matrix
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = round(np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)])))
                dis_mat[i][j] = tmp
        return dis_mat

    # compute the path length
    def compute_pathlen(self, path, dis_mat):
        try:
            a = path[0]
            b = path[-1]
        except:
            import pdb
            pdb.set_trace()
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # compute the fitness value
    def compute_adp(self, offsprings):
        adp = []
        for offspring in offsprings:
            if isinstance(offspring, int):
                import pdb
                pdb.set_trace()
            length = self.compute_pathlen(offspring, self.dis_mat)
            adp.append(self.optimal_length / length)
        return np.array(adp)

    # crossover function
    def ga_cross(self, x, y):
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        order = list(random.sample(path_list, 2))
        order.sort()
        start, end = order

        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = y.index(sub)
            if not (index >= start and index < end):
                x_conflict_index.append(index)

        y_confict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = x.index(sub)
            if not (index >= start and index < end):
                y_confict_index.append(index)

        assert len(x_conflict_index) == len(y_confict_index)

        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp

        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            y[i], x[j] = x[j], y[i]

        assert len(set(x)) == len_ and len(set(y)) == len_
        return list(x), list(y)

    # select how many parent to keep for next population
    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.offsprings[index])
            parents_score.append(scores[index])
        return parents, parents_score

    # selection function
    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * self.optimal_length / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    # mutation function
    def ga_mutate(self, gene):
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        np.random.shuffle(tmp)
        # tmp = tmp[::-1]
        gene[start:end] = tmp
        return list(gene)

    # GA cycle
    def ga(self):
        # calculate the fitness value
        scores = self.compute_adp(self.offsprings)

        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        offsprings = parents.copy()

        num_total = self.num_total
        self.fuzzy_system.input['fitness'] = tmp_best_score
        self.fuzzy_system.compute()
        num_total = self.fuzzy_system.output['num_population']
        # print(num_total)
        mutate_rate = self.fuzzy_system.output['mutate_rate']
        # print(mutate_rate)
        crossover_rate = self.fuzzy_system.output['crossover_rate']
        # print(crossover_rate)
        # next population
        while len(offsprings) < num_total:
            # selection
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            # crossover
            if np.random.rand() < crossover_rate:
                gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            else:
                gene_x_new, gene_y_new = gene_x, gene_y
            # mutation
            if np.random.rand() < mutate_rate:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < mutate_rate:
                gene_y_new = self.ga_mutate(gene_y_new)
            
            if not gene_x_new in offsprings:
                offsprings.append(gene_x_new)
            if not gene_y_new in offsprings:
                offsprings.append(gene_y_new)

        self.offsprings = offsprings

        return tmp_best_one, tmp_best_score

    # run the GA cycle
    def run(self):
        BEST_LIST = None
        best_score = -math.inf
        self.best_record = []
        for i in range(1, self.iteration + 1):
            tmp_best_one, tmp_best_score = self.ga()
            self.iter_x.append(i)
            self.iter_y.append(self.optimal_length / tmp_best_score)
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
            self.best_record.append(self.optimal_length / best_score)
            print(i, int(self.optimal_length / best_score))
        print(int(self.optimal_length / best_score))
        return self.location[BEST_LIST], self.optimal_length / best_score


# read the tsp file
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data


# main function
if __name__ == "__main__":

    # read input
    data = read_tsp(r'') # enter your target file directory
    data = np.array(data)
    data = data[:, 1:]

    # run the algorithm
    Best, Best_path = math.inf, None
    model = GA(num_city=data.shape[0], num_total=25, iteration=500, data=data.copy())
    Best_path, Best = model.run()

    # plot the result
    plt.scatter(Best_path[:, 1], Best_path[:, 0])
    Best_path = np.vstack([Best_path, Best_path[0]])
    plt.plot(Best_path[:, 1], Best_path[:, 0])
    plt.title('Tour route')
    plt.show()

    iterations = range(model.iteration)
    best_record = model.best_record
    plt.plot(iterations, best_record)
    plt.title('Tour length')
    plt.show()


