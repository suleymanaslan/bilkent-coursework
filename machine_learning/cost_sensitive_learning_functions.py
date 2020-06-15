import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_baseline(data_train_x, data_train_y, data_test_x, data_test_y):
    clf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=550)
    clf.fit(data_train_x, data_train_y)
    
    train_acc = (clf.predict(data_train_x) == data_train_y).mean()
    test_acc = (clf.predict(data_test_x) == data_test_y).mean()
    
    train_acc_c1 = (clf.predict(data_train_x[data_train_y == 1]) == data_train_y[data_train_y == 1]).mean()
    test_acc_c1 = (clf.predict(data_test_x[data_test_y == 1]) == data_test_y[data_test_y == 1]).mean()
    
    train_acc_c2 = (clf.predict(data_train_x[data_train_y == 2]) == data_train_y[data_train_y == 2]).mean()
    test_acc_c2 = (clf.predict(data_test_x[data_test_y == 2]) == data_test_y[data_test_y == 2]).mean()
    
    train_acc_c3 = (clf.predict(data_train_x[data_train_y == 3]) == data_train_y[data_train_y == 3]).mean()
    test_acc_c3 = (clf.predict(data_test_x[data_test_y == 3]) == data_test_y[data_test_y == 3]).mean()
    
    print(f"Train Acc:{train_acc:.4f}")
    print(f"Test Acc:{test_acc:.4f}")
    
    print(f"Class 1 Train Acc:{train_acc_c1:.4f}")
    print(f"Class 1 Test Acc:{test_acc_c1:.4f}")
    
    print(f"Class 2 Train Acc:{train_acc_c2:.4f}")
    print(f"Class 2 Test Acc:{test_acc_c2:.4f}")
    
    print(f"Class 3 Train Acc:{train_acc_c3:.4f}")
    print(f"Class 3 Test Acc:{test_acc_c3:.4f}")
    
    train_misclassification = (clf.predict(data_train_x) != data_train_y).sum()
    print(f"Train Misclassifications:{train_misclassification}")


def forward_selection(data_train_x, data_train_y, data_cost):
    clf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=550)
    cur_features = []
    cur_joint_cost = np.inf
    cur_feature_cost = 0
    done = False

    acc_0 = (np.ones_like(data_train_y) * 3 == data_train_y).mean()
    all_f_cost = data_cost.sum()

    while not done:
        new_feature_index = -1
        for i in range(data_train_x.shape[1]):
            if i in cur_features:
                continue

            used_features = []
            current_cost = cur_feature_cost
            if i == 20:
                used_features.append(i-2)
                used_features.append(i-1)
                current_cost += data_cost[i-2]
                current_cost += data_cost[i-1]
            else:
                current_cost += data_cost[i]
            used_features.append(i)

            all_features = cur_features + used_features

            used_features_x = None
            for u_feature in all_features:
                if used_features_x is None:
                    used_features_x = np.expand_dims(data_train_x[:,u_feature], axis=1)
                else:
                    used_features_x = np.append(used_features_x, np.expand_dims(data_train_x[:,u_feature], axis=1), axis=1)

            clf.fit(used_features_x, data_train_y)

            used_features_acc = (clf.predict(used_features_x) == data_train_y).mean()
            used_features_acc_c1 = (clf.predict(used_features_x[data_train_y == 1]) == data_train_y[data_train_y == 1]).mean()
            used_features_acc_c2 = (clf.predict(used_features_x[data_train_y == 2]) == data_train_y[data_train_y == 2]).mean()
            used_features_acc_c3 = (clf.predict(used_features_x[data_train_y == 3]) == data_train_y[data_train_y == 3]).mean()

            joint_cost = (acc_0 + (1 - acc_0) * (current_cost / all_f_cost)) / used_features_acc

            if joint_cost < cur_joint_cost:
                cur_joint_cost = joint_cost
                new_feature_index = i
                best_feature_cost = current_cost
                best_features = used_features
                best_acc = used_features_acc
                best_acc_c1 = used_features_acc_c1
                best_acc_c2 = used_features_acc_c2
                best_acc_c3 = used_features_acc_c3

            print(f"Features:{used_features},\tAcc:{used_features_acc:.2f}, Class1_Acc:{used_features_acc_c1:.4f}, "
                  f"Class2_Acc:{used_features_acc_c2:.4f}, Class3_Acc:{used_features_acc_c3:.4f}, "
                  f"Joint Cost:{joint_cost:.4f}, Feature Cost:{current_cost}")

        if new_feature_index > -1:
            cur_features = cur_features + best_features
            cur_feature_cost = best_feature_cost
        else:
            done = True
            print("Done")
        print(f"Selected Features:{cur_features}, Acc:{best_acc:.4f}, Class1_Acc:{best_acc_c1:.4f}, Class2_Acc:{best_acc_c2:.4f}, "
              f"Class3_Acc:{best_acc_c3:.4f}, Joint Cost:{cur_joint_cost:.4f}, Feature Cost:{cur_feature_cost}")
        print("")
    return cur_features


def evaluate_forward_selection(cur_features, data_train_x, data_test_x, data_train_y, data_test_y):
    clf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=550)
    used_features_x = None
    for u_feature in cur_features:
        if used_features_x is None:
            used_features_x = np.expand_dims(data_train_x[:,u_feature], axis=1)
        else:
            used_features_x = np.append(used_features_x, np.expand_dims(data_train_x[:,u_feature], axis=1), axis=1)

    used_features_test_x = None
    for u_feature in cur_features:
        if used_features_test_x is None:
            used_features_test_x = np.expand_dims(data_test_x[:,u_feature], axis=1)
        else:
            used_features_test_x = np.append(used_features_test_x, np.expand_dims(data_test_x[:,u_feature], axis=1), axis=1)

    clf.fit(used_features_x, data_train_y)

    selected_features_train_acc = (clf.predict(used_features_x) == data_train_y).mean()
    selected_features_train_acc_c1 = (clf.predict(used_features_x[data_train_y == 1]) == data_train_y[data_train_y == 1]).mean()
    selected_features_train_acc_c2 = (clf.predict(used_features_x[data_train_y == 2]) == data_train_y[data_train_y == 2]).mean()
    selected_features_train_acc_c3 = (clf.predict(used_features_x[data_train_y == 3]) == data_train_y[data_train_y == 3]).mean()

    selected_features_test_acc = (clf.predict(used_features_test_x) == data_test_y).mean()
    selected_features_test_acc_c1 = (clf.predict(used_features_test_x[data_test_y == 1]) == data_test_y[data_test_y == 1]).mean()
    selected_features_test_acc_c2 = (clf.predict(used_features_test_x[data_test_y == 2]) == data_test_y[data_test_y == 2]).mean()
    selected_features_test_acc_c3 = (clf.predict(used_features_test_x[data_test_y == 3]) == data_test_y[data_test_y == 3]).mean()

    print(f"Selected Features:{cur_features}, "
          f"Train Acc:{selected_features_train_acc:.4f}, "
          f"Train Class1_Acc:{selected_features_train_acc_c1:.4f}, "
          f"Train Class2_Acc:{selected_features_train_acc_c2:.4f}, "
          f"Train Class3_Acc:{selected_features_train_acc_c3:.4f}, ")

    print(f"Selected Features:{cur_features}, "
          f"Test Acc:{selected_features_test_acc:.4f}, "
          f"Test Class1_Acc:{selected_features_test_acc_c1:.4f}, "
          f"Test Class2_Acc:{selected_features_test_acc_c2:.4f}, "
          f"Test Class3_Acc:{selected_features_test_acc_c3:.4f}, ")


def calculate_fitness(population, data_train_x, data_train_y, data_cost):
    clf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=550)
    
    acc_0 = (np.ones_like(data_train_y) * 3 == data_train_y).mean()
    all_f_cost = data_cost.sum()
    
    fitness_values = []
    for individual in population:
        cur_features = []
        cur_feature_cost = 0
        for u_feature, is_used in enumerate(individual):
            if is_used:
                cur_features.append(u_feature)
                cur_feature_cost += data_cost[u_feature]
        if 18 in cur_features and 19 in cur_features:
            cur_features.append(20)
        
        used_features_x = None
        for u_feature in cur_features:
            if used_features_x is None:
                used_features_x = np.expand_dims(data_train_x[:,u_feature], axis=1)
            else:
                used_features_x = np.append(used_features_x, np.expand_dims(data_train_x[:,u_feature], axis=1), axis=1)
        
        clf.fit(used_features_x, data_train_y)
        
        used_features_acc = (clf.predict(used_features_x) == data_train_y).mean()
        
        joint_cost = (acc_0 + (1 - acc_0) * (cur_feature_cost / all_f_cost)) / used_features_acc
        
        fitness_values.append(joint_cost)
    
    return fitness_values


def selection(population, fitness_values, to_be_selected):
    new_population = []
    temp_fitness = fitness_values.copy()
    for i in range(to_be_selected):
        best_fit = np.argmin(temp_fitness)
        temp_fitness[best_fit] = np.inf
        new_population.append(population[best_fit])
    return new_population


def cross_over(new_population, to_be_crossed_over):
    children = None
    for _ in range(to_be_crossed_over):
        temp = np.arange(len(new_population))
        first_parent_id = np.random.choice(temp)
        temp = np.delete(temp, first_parent_id)
        second_parent_id = np.random.choice(temp)
        first_parent = new_population[first_parent_id]
        second_parent = new_population[second_parent_id]
        mask = np.random.randint(2, size=(20))
        first_genes = np.logical_and(first_parent, mask == 0).astype(np.int32)
        second_genes = np.logical_and(second_parent, mask == 1).astype(np.int32)
        child = np.logical_or(first_genes, second_genes).astype(np.int32)
        children = child if children is None else np.vstack((children, child))
        first_genes = np.logical_and(first_parent, mask == 1).astype(np.int32)
        second_genes = np.logical_and(second_parent, mask == 0).astype(np.int32)
        child = np.logical_or(first_genes, second_genes).astype(np.int32)
        children = child if children is None else np.vstack((children, child))
    return children
    print(children)


def mutate(individuals_to_mutate):
    mutants = None
    nb_of_ones = np.random.randint(2) + 1
    nb_of_zeros = 20 - nb_of_ones
    mutation_mask = np.array([0] * nb_of_zeros + [1] * nb_of_ones)
    np.random.shuffle(mutation_mask)
    for ix, individual in enumerate(individuals_to_mutate):
        individual_inverse = 1-individual
        normal_genes = np.logical_and(individual, mutation_mask == 0).astype(np.int32)
        mutant_genes = np.logical_and(individual_inverse, mutation_mask == 1).astype(np.int32)
        mutant = np.logical_or(normal_genes, mutant_genes).astype(np.int32)
        mutants = mutant if mutants is None else np.vstack((mutants, mutant))
    return mutants


def next_generation(population, fitness_values, to_be_selected, to_be_crossed_over, to_be_mutated):
    new_population = selection(population, fitness_values, to_be_selected)
    children = cross_over(new_population, to_be_crossed_over)
    mutants = mutate(children[:to_be_mutated])
    children[:to_be_mutated] = mutants
    return np.vstack((new_population, children))


def eval_individual(individual, data_train_x, data_train_y, data_cost, train=True):
    clf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=550)
    acc_0 = (np.ones_like(data_train_y) * 3 == data_train_y).mean()
    all_f_cost = data_cost.sum()
    cur_features = []
    cur_feature_cost = 0
    for u_feature, is_used in enumerate(individual):
        if is_used:
            cur_features.append(u_feature)
            cur_feature_cost += data_cost[u_feature]
    if 18 in cur_features and 19 in cur_features:
        cur_features.append(20)
    
    used_features_x = None
    for u_feature in cur_features:
        if used_features_x is None:
            used_features_x = np.expand_dims(data_train_x[:,u_feature], axis=1)
        else:
            used_features_x = np.append(used_features_x, np.expand_dims(data_train_x[:,u_feature], axis=1), axis=1)
    
    clf.fit(used_features_x, data_train_y)

    used_features_acc = (clf.predict(used_features_x) == data_train_y).mean()
    used_features_acc_c1 = (clf.predict(used_features_x[data_train_y == 1]) == data_train_y[data_train_y == 1]).mean()
    used_features_acc_c2 = (clf.predict(used_features_x[data_train_y == 2]) == data_train_y[data_train_y == 2]).mean()
    used_features_acc_c3 = (clf.predict(used_features_x[data_train_y == 3]) == data_train_y[data_train_y == 3]).mean()

    joint_cost = (acc_0 + (1 - acc_0) * (cur_feature_cost / all_f_cost)) / used_features_acc
    
    print(f"Features:{cur_features},\tAcc:{used_features_acc:.4f}, Class1_Acc:{used_features_acc_c1:.4f}, "
          f"Class2_Acc:{used_features_acc_c2:.4f}, Class3_Acc:{used_features_acc_c3:.4f}, "
          f"Joint Cost:{joint_cost:.4f}, Feature Cost:{cur_feature_cost}")


def genetic_algorithm(data_train_x, data_train_y, data_cost):
    p = 80
    r = 0.8
    m = 0.6
    to_be_selected = round((1 - r) * p)
    to_be_crossed_over = round(r * p / 2)
    to_be_mutated = round(m * p)
    initial_features = 5
    
    final_individual = None
    final_joint_cost = np.inf
    done = False
    for g in range(10):
        if g == 0:
            population = None
            for _ in range(p):
                row_p = np.array([0] * (20 - initial_features) + [1] * initial_features)
                np.random.shuffle(row_p)
                population = row_p if population is None else np.vstack((population, row_p))
            fitness_values = calculate_fitness(population, data_train_x, data_train_y, data_cost)
        else:
            population = next_generation(population, fitness_values, to_be_selected, to_be_crossed_over, to_be_mutated)
            fitness_values = calculate_fitness(population, data_train_x, data_train_y, data_cost)
        best_fit_ix = np.argsort(fitness_values)[0]
        best_fit_individual = population[best_fit_ix]
        best_fit_cost = fitness_values[best_fit_ix]
        print(f"Generation:{g}, Individual:{best_fit_individual}, Joint Cost:{best_fit_cost:.8f}")
        eval_individual(best_fit_individual, data_train_x, data_train_y, data_cost)
        
        if best_fit_cost < final_joint_cost:
            final_individual = best_fit_individual
            final_joint_cost = best_fit_cost
        else:
            done = True
        if done:
            break
    return best_fit_individual


def evaulate_genetic_algorithm(best_fit_individual, data_train_x, data_train_y, data_test_x, data_test_y, data_cost):
    clf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=550)
    cur_features = []
    cur_feature_cost = 0
    for u_feature, is_used in enumerate(best_fit_individual):
        if is_used:
            cur_features.append(u_feature)
            cur_feature_cost += data_cost[u_feature]
    if 18 in cur_features and 19 in cur_features:
        cur_features.append(20)
    
    used_features_x = None
    for u_feature in cur_features:
        if used_features_x is None:
            used_features_x = np.expand_dims(data_train_x[:,u_feature], axis=1)
        else:
            used_features_x = np.append(used_features_x, np.expand_dims(data_train_x[:,u_feature], axis=1), axis=1)
    
    used_features_test_x = None
    for u_feature in cur_features:
        if used_features_test_x is None:
            used_features_test_x = np.expand_dims(data_test_x[:,u_feature], axis=1)
        else:
            used_features_test_x = np.append(used_features_test_x, np.expand_dims(data_test_x[:,u_feature], axis=1), axis=1)
    
    clf.fit(used_features_x, data_train_y)
    
    selected_features_train_acc = (clf.predict(used_features_x) == data_train_y).mean()
    selected_features_train_acc_c1 = (clf.predict(used_features_x[data_train_y == 1]) == data_train_y[data_train_y == 1]).mean()
    selected_features_train_acc_c2 = (clf.predict(used_features_x[data_train_y == 2]) == data_train_y[data_train_y == 2]).mean()
    selected_features_train_acc_c3 = (clf.predict(used_features_x[data_train_y == 3]) == data_train_y[data_train_y == 3]).mean()
    
    selected_features_test_acc = (clf.predict(used_features_test_x) == data_test_y).mean()
    selected_features_test_acc_c1 = (clf.predict(used_features_test_x[data_test_y == 1]) == data_test_y[data_test_y == 1]).mean()
    selected_features_test_acc_c2 = (clf.predict(used_features_test_x[data_test_y == 2]) == data_test_y[data_test_y == 2]).mean()
    selected_features_test_acc_c3 = (clf.predict(used_features_test_x[data_test_y == 3]) == data_test_y[data_test_y == 3]).mean()
    
    print(f"Selected Features:{cur_features}, "
          f"Train Acc:{selected_features_train_acc:.4f}, "
          f"Train Class1_Acc:{selected_features_train_acc_c1:.4f}, "
          f"Train Class2_Acc:{selected_features_train_acc_c2:.4f}, "
          f"Train Class3_Acc:{selected_features_train_acc_c3:.4f}, ")
    
    print(f"Selected Features:{cur_features}, "
          f"Test Acc:{selected_features_test_acc:.4f}, "
          f"Test Class1_Acc:{selected_features_test_acc_c1:.4f}, "
          f"Test Class2_Acc:{selected_features_test_acc_c2:.4f}, "
          f"Test Class3_Acc:{selected_features_test_acc_c3:.4f}, ")
