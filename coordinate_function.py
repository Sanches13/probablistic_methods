import re, copy
from itertools import product
import math
from pprint import pprint

class CoordinateFunction:
    def __init__(self, substitute: list[int], bit_number: int, power: int):
        self.bit_number = bit_number
        self.power = power
        self.substitute = substitute
        self.vector: list[int] = []
        self.weight = 0
        self.zhegalkin_polynomial: list[int] = []
        self.zhegalkin_polynomial_string = ""
        self.predominance: float = 0
        self.truth_table: dict[tuple[int], int] = dict()
        self.WA_coefs: list[float] = []
        self.correlation_immunity = 0
        self.elasticity = 0
        self.coefs_stat_structure: list[float] = []
        self.best_linear_approximation: list[tuple[int]] = []
        self.autocorrelation_coefs: list[float] = []
        self.bent_status: bool = False
        self.zapret: list[int] = []
        self.strong_equiprobability: bool = False

    def compute_coordinate_function(self) -> None:
        for value in self.substitute:
            bin_value = bin(value)[2:].zfill(self.power)[::-1]
            self.vector.append(int(bin_value[self.bit_number]))
        
    def compute_weight(self) -> None:
        self.weight = sum(self.vector)

    # Нужно для Л1 П4
    def compute_zhegalkin_polynomial(self) -> None:
        vector_len = len(self.vector)
        pascal_triangle: list[list[int]] = [[0 for j in range(vector_len)] for i in range(vector_len)]
        for i in range(1, vector_len + 1):
            pascal_triangle[i - 1][0] = self.vector[vector_len - i]
            for j in range(1, i):
                pascal_triangle[i - 1][j] = pascal_triangle[i - 2][j - 1] ^ pascal_triangle[i - 1][j - 1]
        self.zhegalkin_polynomial = pascal_triangle[vector_len - 1]

    # Строковое представление многочлена Жегалкина
    def get_str_ZP(self) -> None:
        if self.zhegalkin_polynomial[0] == 1:
           self.zhegalkin_polynomial_string += "1 ⊕  "

        for i in range(1, len(self.zhegalkin_polynomial)):
            if self.zhegalkin_polynomial[i] == 1:
                bin_value = bin(i)[2:].zfill(self.power)
                for coef, bit in enumerate(bin_value):
                    if int(bit) == 1:
                        self.zhegalkin_polynomial_string += f"x_{coef + 1} "
                self.zhegalkin_polynomial_string += "⊕  "

        self.zhegalkin_polynomial_string = self.zhegalkin_polynomial_string[:-4]
    
    # Л1 П5
    def get_fictive_terms(self) -> list[str]:
        # ["x_1": False, "x_2": False, ...]
        terms: list[str] = {f"x_{i}" for i in range(1, self.power + 1)}
        fictive_terms = []
        for term in terms:
            if not re.search(term, self.zhegalkin_polynomial_string):
                fictive_terms.append(term)
        return fictive_terms
    
    # Л2 П1
    def compute_predominance(self) -> None:
        self.predominance = 1 - (self.weight) / (2 ** (self.power - 1))
    
    def generate_truth_table(self) -> None:
        for i in range(len(self.vector)):
            bin_value = bin(i)[2:].zfill(self.power)
            key = tuple([int(bit) for bit in bin_value])
            self.truth_table[key] = self.vector[i]

    # def compute_zapret(self) -> None:
    #     self.generate_truth_table()
    #     zapret = []

    #     # init current truth table
    #     current_truth_table = copy.deepcopy(self.truth_table)
    #     for key, value in current_truth_table.items():
    #         current_truth_table[key] = tuple([value])
        
    #     # start compute zapret
    #     while True:
    #         next_truth_table: dict[tuple[int], tuple[int]] = dict()
    #         for combination, value in current_truth_table.items():
    #             for i in [0, 1]:
    #                 extended_combination = tuple(list(combination) + [i])
    #                 extended_key = tuple(list(value) + [self.truth_table[extended_combination[len(extended_combination) - self.power:]]])
    #                 next_truth_table[extended_combination] = extended_key
    #         # pprint(next_truth_table)

    #         # Считаем частоту значений
    #         value_frequency = {}    
    #         for value in next_truth_table.values():
    #             value_frequency[value] = value_frequency.get(value, 0) + 1

    #         if len(value_frequency.values()) == 1:
    #             for value in next_truth_table.values():
    #                 zapret = list(value)
    #                 break
    #             break
            
    #         # Находим наименьшую частоту
    #         min_frequency = min(value_frequency.values())
    #         min_frequency_value = None
    #         for value, frequency in value_frequency.items():
    #             if frequency == min_frequency:
    #                 min_frequency_value = value
    #                 break
            
    #         current_truth_table: dict[tuple[int], tuple[int]] = dict()
    #         for key, value in next_truth_table.items():
    #             if value == min_frequency_value:
    #                 current_truth_table[key] = value

    #     # invert last value
    #     zapret[-1] = (zapret[-1] + 1) % 2
    #     self.zapret = zapret
    def compute_zapret(self) -> None:
        class Tree:
            def __init__(self, vec, power):
                self.vectors = vec.copy()
                self.parent = self
                self.c = []
                self.power = power
                
            def next_step(self, func):
                self.zero = Tree([], self.power)
                self.one = Tree([], self.power)
                self.zero.parent = self
                self.one.parent = self
                self.zero.c = self.c + [0]
                self.one.c = self.c + [1]
                for i in self.vectors:
                    j = i[-(self.power-1):]
                    sum = 0
                    for k in range(len(j)):
                        sum = sum + j[k] * (2**(self.power-k-1))
                    if (func[sum]):
                        self.one.vectors.append(i + [0])
                    else:
                        self.zero.vectors.append(i + [0])
                    
                    sum = sum + 1
                    if (func[sum]):
                        self.one.vectors.append(i + [1])
                    else:
                        self.zero.vectors.append(i + [1])
                                    
        vec = product([0, 1], repeat = self.power-1)
        v = [list(i) for i in vec]
        t = Tree(v, self.power)
        zapret = [0 for i in range(self.power)]

        tmp = [t]
        min = math.inf
        next = []
        count = 0
        while(1):
            for i in tmp:
                i.next_step(self.vector)
                if (len(i.one.vectors) < min):
                    min = len(i.one.vectors)
                    next = [i.one]
                elif (len(i.one.vectors) == min):
                    next.append(i.one)
                    
                if (len(i.zero.vectors) < min):
                    min = len(i.zero.vectors)
                    next = [i.zero]
                elif (len(i.zero.vectors) == min):
                    next.append(i.zero)
            if (min == 0):
                break
            tmp = next.copy()
            next = []
            count = count + 1
            if (min == 2**(self.power) and count > self.power*4):
                zapret = -1
                break
        for i in tmp:
            i.next_step(self.vector)
            if (len(i.one.vectors) == 0):
                zapret = i.one.c
            if (len(i.zero.vectors) == 0):
                zapret = i.zero.c

        self.zapret = zapret

    def define_strong_equiprobability(self):
        if self.zapret != []:
            self.strong_equiprobability = False
    
    # Вычисляем вектор БПФ
    def _compute_BPF(self) -> list[int]:
        vector = copy.deepcopy(self.vector)
        for step in range(self.power):
            next_step_vector: list[int] = [0 for i in range(len(vector))]
            pairs: list[list[int]] = []
            visited_indices: list[int] = []
            for i in range(len(self.vector)):
                if i not in visited_indices:
                    pairs.append([i, i + 2 ** step])
                    visited_indices.append(i)
                    visited_indices.append(i + 2 ** step)
            for pair in pairs:
                next_step_vector[pair[0]] = vector[pair[0]] + vector[pair[1]]
                next_step_vector[pair[1]] = vector[pair[0]] - vector[pair[1]]
            vector = copy.deepcopy(next_step_vector)
        return vector

    def compute_WA_coefs(self) -> list[float]:
        bpf_vector = self._compute_BPF()
        for i in range(len(bpf_vector)):
            self.WA_coefs.append(-1 / 2 ** (self.power-1) * bpf_vector[i])
        self.WA_coefs[0] += 1
        # костыль
        for i in range(len(self.WA_coefs)):
            if self.WA_coefs[i] == -0.0:
                self.WA_coefs[i] = 0.0

    def _get_values_by_weight(self, vector_weight: int) -> list[tuple[int]]:
        result = []
        for vector_values in self.truth_table.keys():
            if sum(vector_values) == vector_weight:
                result.append(vector_values)
        return result
    
    def compute_correlation_immunity(self) -> None:
        for vector_weight in range(1, self.power + 1):
            vector_values = self._get_values_by_weight(vector_weight)
            for vector_value in vector_values:
                index_as_str = ''.join(map(str, vector_value))
                index = int(index_as_str, 2)
                if self.WA_coefs[index] != 0:
                    return
            self.correlation_immunity = vector_weight

    # Для общего случая надо допилить, но нет понимания как
    # Для нашего случая отработает, т.к. сбалансированность есть 
    def compute_elasticity(self) -> None:
        if self.predominance == 0:
            self.elasticity = self.correlation_immunity
        else:
            print("Функция не сбалансирована, что делать в это случае - хз")
            self.elasticity = None

    def compute_coefs_stat_structure(self) -> None:
        for WA_coef in self.WA_coefs:
            self.coefs_stat_structure.append(WA_coef * 2 ** (self.power - 1))

    def compute_best_linear_approximation(self) -> None:
        # get max coef
        max_coef_stat_structure = max(self.coefs_stat_structure)
        for i, vector_value in enumerate(self.truth_table):
            if self.coefs_stat_structure[i] == max_coef_stat_structure:
                self.best_linear_approximation.append(vector_value)

    def compute_autocorrelation_coefs(self) -> None:
        self.autocorrelation_coefs = [0 for i in range(2 ** self.power)]
        for i, u_vector in enumerate(self.truth_table):
            for x_vector in self.truth_table:
                u_xor_x = tuple([u_vector[j] ^ x_vector[j] for j in range(self.power)])
                self.autocorrelation_coefs[i] += (-1) ** (self.truth_table[x_vector] ^ self.truth_table[u_xor_x])
            self.autocorrelation_coefs[i] /= 2 ** self.power

    def check_bent_status(self) -> None:
        if self.power % 2 == 0:
            for i in range(len(self.WA_coefs) - 1):
                if abs(self.WA_coefs[i]) != abs(self.WA_coefs[i + 1]):
                    return False
            return True
        return False