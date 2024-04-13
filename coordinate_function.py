import re, copy
from itertools import combinations
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

    # # Нужно для вывода многочлена Жегалкина
    # def _find_combinations(self, arr, k):
    #     result = []
    #     n = len(arr)
        
    #     # Вспомогательная функция для поиска комбинаций
    #     def backtrack(start, combination):
    #         if len(combination) == k:
    #             result.append(combination[:])
    #             return
    #         for i in range(start, n):
    #             combination.append(arr[i])
    #             backtrack(i + 1, combination)
    #             combination.pop()

    #     backtrack(0, [])
    #     return result

    # # Нужно для вывода многочлена Жегалкина
    # def _get_terms(self) -> list[list[str]]:
    #     ZP = []
    #     ZP.append(1)  # начальное значение
        
    #     terms = [f"x_{i}" for i in range(1, self.power + 1)]
    #     for i in range(1, self.power + 1):
    #         combinations = self._find_combinations(terms, i)
    #         for el in combinations:
    #             # Произведение переменных в текущей комбинации
    #             product = ' '.join(el)
    #             ZP.append(product)
    #     return ZP

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
    
    # # Нужно для Л2 П2, генерирует возможные значения x_1,...,x_n
    # def _generate_binary_combinations(self):
    #     def generate_combinations_helper(current_combination, index):
    #         if index == self.power:
    #             yield current_combination
    #             return
    #         current_combination[index] = 0
    #         yield from generate_combinations_helper(current_combination, index + 1)
    #         current_combination[index] = 1
    #         yield from generate_combinations_helper(current_combination, index + 1)

    #     initial_combination = [0] * self.power
    #     yield from generate_combinations_helper(initial_combination, 0)

    # def _compute_function_value(self, current_combination: list[int], coeffs: list[int]) -> int:
    #     zp = [1]  # initial state

    #     for i in range(1, len(current_combination) + 1):
    #         combs = combinations(current_combination, i)
    #         for el in combs:
    #             product = 1
    #             for num in el:
    #                 product *= num
    #             zp.append(product)

    #     for i in range(2 ** self.power):
    #         zp[i] *= coeffs[i]

    #     return zp.count(1) % 2

    # def generate_truth_table(self) -> dict[tuple[int], int]:
    #     truth_table: dict[tuple[int], int] = dict()
    #     for terms_value in self._generate_binary_combinations():
    #         print(terms_value)
    #         truth_table[tuple(terms_value)] = self._compute_function_value(terms_value, self.zhegalkin_polynomial)
    #     return truth_table
    
    def generate_truth_table(self) -> None:
        for i in range(len(self.vector)):
            bin_value = bin(i)[2:].zfill(self.power)
            key = tuple([int(bit) for bit in bin_value])
            self.truth_table[key] = self.vector[i]

    def compute_zapret(self) -> list[int]:
        self.generate_truth_table()
        zapret = []

        # init current truth table
        current_truth_table = copy.deepcopy(self.truth_table)
        for key, value in current_truth_table.items():
            current_truth_table[key] = tuple([value])
        
        # start compute zapret
        # while len(current_truth_table.values()) != 1:
        while True:
            next_truth_table: dict[tuple[int], tuple[int]] = dict()
            for combination, value in current_truth_table.items():
                for i in [0, 1]:
                    extended_combination = tuple(list(combination) + [i])
                    extended_key = tuple(list(value) + [self.truth_table[extended_combination[len(extended_combination) - self.power:]]])
                    next_truth_table[extended_combination] = extended_key

            # Считаем частоту значений
            value_frequency = {}    
            for value in next_truth_table.values():
                value_frequency[value] = value_frequency.get(value, 0) + 1
            # pprint(value_frequency)

            if len(value_frequency.values()) == 1:
                for value in next_truth_table.values():
                    zapret = list(value)
                    break
                break
            
            # Находим наименьшую частоту
            min_frequency = min(value_frequency.values())
            min_frequency_value = None
            for value, frequency in value_frequency.items():
                if frequency == min_frequency:
                    min_frequency_value = value
                    break
            
            current_truth_table: dict[tuple[int], tuple[int]] = dict()
            for key, value in next_truth_table.items():
                if value == min_frequency_value:
                    current_truth_table[key] = value

        # invert last value
        zapret[-1] = (zapret[-1] + 1) % 2
        return zapret
    
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