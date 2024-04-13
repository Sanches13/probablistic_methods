from substitute import Substitute
from coordinate_function import CoordinateFunction

POWER = 6
ELEMENTS_COUNT = 2 ** POWER

def main() -> None:
    # test_vector = [28, 60, 15, 2, 4, 53, 33, 22, 56, 30, 21, 38, 41, 20, 47, 32, 26, 46, 59, 35, 48, 23, 61, 18, 37, 51, 42, 50, 44, 52, 24, 5, 16, 43, 40, 34, 0, 11, 9, 63, 3, 8, 27, 17, 55, 54, 49, 7, 6, 13, 58, 14, 57, 31, 45, 29, 19, 39, 36, 10, 25, 62, 12, 1]
    # POWER = 4
    # test_vector = [3, 5, 15, 12, 8, 0, 4, 14, 10, 6, 1, 11, 9, 13, 2, 7]
    # POWER = 3
    # test_vector = [6, 4, 3, 1, 0, 2, 5, 7]
    # test_vector = [0, 0, 0, 7, 0, 7, 7, 7]
    # substitute = Substitute(POWER, test_vector)
    substitute = Substitute(POWER)
    print(str.center("Лабораторная работа №1", 100, "="))
    print("№1. Сгенерировать случайную подстановку. Случайная подстановка:")
    print(substitute.vector)
    print("№2. Построение векторов значений координатных функций.")
    for i, coordinate_function in enumerate(substitute.coordinate_functions):
        coordinate_function.compute_coordinate_function()
        print(f"f_{i + 1} = " + str(coordinate_function.vector))
        # print(coordinate_function.vector)
    print("№3. Вычисление веса координатных функций.")
    for i, coordinate_function in enumerate(substitute.coordinate_functions):
        coordinate_function.compute_weight()
        print(f"w(f_{i + 1}) = " + str(coordinate_function.weight))
    print("№4. Многочлены Жегалкина координатных функций.")
    for i, coordinate_function in enumerate(substitute.coordinate_functions):
        coordinate_function.compute_zhegalkin_polynomial()
        print(coordinate_function.zhegalkin_polynomial)
        coordinate_function.get_str_ZP()
        print(f"ZP(f_{i + 1}) = " + str(coordinate_function.zhegalkin_polynomial_string))
    print("№5. Найти фиктивные переменные координатных функций.")
    for i, coordinate_function in enumerate(substitute.coordinate_functions):
        print(f"Фиктивные переменные функции f_{i + 1}: " + str(coordinate_function.get_fictive_terms()))

    print(str.center("Лабораторная работа №2", 100, "="))
    print("№1. Преобладание нулей над единицами")
    for i, coordinate_function in enumerate(substitute.coordinate_functions):
        coordinate_function.compute_predominance()
        print(f"delta(f_{i + 1}) = " + str(coordinate_function.predominance))

    # print("№2. Построение запретов")
    # for i, coordinate_function in enumerate(substitute.coordinate_functions):
    #     print(f"zapret(f_{i + 1}) = " + str(coordinate_function.compute_zapret()))
    # print("№3. Проверка на равновероятность.")
    # Если нет запретов, то сильно равновероятна, иначе нет
        
    print(str.center("Лабораторная работа №3", 100, "="))
    print("№1. Корреляционная иммунность и эластичность функций")
    for i, coordinate_function in enumerate(substitute.coordinate_functions):
        coordinate_function.compute_WA_coefs()
        coordinate_function.generate_truth_table()
        print(f"Для f_{i + 1} коэффициенты Уолша-Адамара следующие:" + str(coordinate_function.WA_coefs))
        coordinate_function.compute_correlation_immunity()
        print(f"Функция f_{i + 1} корреляционно иммуннна порядка {coordinate_function.correlation_immunity}")
        coordinate_function.compute_elasticity()
        print(f"Функция f_{i + 1} эластична порядка {coordinate_function.elasticity}")
    # Перепроверить эластичность, возможно для общего случая
    print("№2. Построение спектра (коэффициентов статистической структуры) и наилучшего линейного приближения.")
    for i, coordinate_function in enumerate(substitute.coordinate_functions):
        coordinate_function.compute_coefs_stat_structure()
        print(f"Спектр функции f({(i + 1)}):")
        print(f"Δ(f_{i + 1}): {coordinate_function.coefs_stat_structure}")
        coordinate_function.compute_best_linear_approximation()
        print(f"Наилучшее линейное приближение для функции f{(i + 1)}:")
        for j, vector in enumerate(coordinate_function.best_linear_approximation):
            str_vector = "⊕ ".join([f"x_{number + 1}" for number, k in enumerate(vector) if k == 1])
            print(f"f_{j + 1}({i + 1}) = {str_vector}")
    print("№3. Коэффициенты автокорреляции")
    for i, coordinate_function in enumerate(substitute.coordinate_functions):
        coordinate_function.compute_autocorrelation_coefs()
        print(f"Cor(f_{i + 1}): {coordinate_function.autocorrelation_coefs}")
    print("№4. Провека на бент-функцию")
    for i, coordinate_function in enumerate(substitute.coordinate_functions):
        coordinate_function.check_bent_status()
        print(f"Является ли функция f_{i + 1} бент-функцией: {coordinate_function.bent_status}")

if __name__ == "__main__":
    main()