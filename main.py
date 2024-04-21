from substitute import Substitute

POWER = 6
ELEMENTS_COUNT = 2 ** POWER

def main() -> None:
    # test_vector = [34, 6, 63, 0, 18, 21, 43, 55, 42, 14, 16, 32, 45, 47, 15, 58, 8, 10, 35, 38, 11, 13, 27, 33, 51, 59, 23, 29, 52, 2, 39, 62, 54, 20, 40, 57, 24, 28, 46, 17, 9, 41, 53, 30, 44, 50, 12, 36, 49, 19, 31, 61, 60, 7, 25, 37, 26, 5, 56, 1, 4, 48, 22, 3]
    # POWER = 4
    # test_vector = [3, 5, 15, 12, 8, 0, 4, 14, 10, 6, 1, 11, 9, 13, 2, 7]
    # POWER = 3
    # test_vector = [6, 4, 3, 1, 0, 2, 5, 7]
    # test_vector = [0, 0, 0, 7, 0, 7, 7, 7]
    # test_vector = [47, 38, 17, 44, 7, 43, 24, 27, 54, 33, 16, 62, 31, 9, 53, 15, 32, 10, 1, 56, 63, 28, 61, 52, 22, 57, 26, 29, 55, 45, 40, 8, 3, 48, 18, 35, 34, 6, 51, 46, 0, 12, 19, 11, 49, 5, 58, 59, 30, 39, 13, 2, 36, 37, 60, 4, 21, 41, 20, 14, 50, 42, 23, 25]
    # substitute = Substitute(POWER, test_vector)
    substitute = Substitute(POWER)
    print(str.center("Лабораторная работа №1", 100, "="))
    print("№1. Сгенерировать случайную подстановку. Случайная подстановка:")
    print(substitute.vector)
    print("№2. Построение векторов значений координатных функций.")
    for i, coordinate_function in enumerate(substitute.coordinate_functions):
        coordinate_function.compute_coordinate_function()
        print(f"f_{i + 1} = " + str(coordinate_function.vector))
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

    print("№2. Построение запретов")
    for i, coordinate_function in enumerate(substitute.coordinate_functions):
        coordinate_function.compute_zapret()
        print(f"zapret(f_{i + 1}) = " + str(coordinate_function.zapret))
    print("№3. Проверка на равновероятность.")
    for i, coordinate_function in enumerate(substitute.coordinate_functions):
        coordinate_function.define_strong_equiprobability()
        print(f"Является ли функция f_{i + 1} сильно равновероятной: {coordinate_function.strong_equiprobability}")
        
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