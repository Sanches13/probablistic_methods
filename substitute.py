import random
from coordinate_function import CoordinateFunction

class Substitute:
    def __init__(self, power: int, vector: list[int] = None) -> None:
        self.power = power
        self.elements_count = 2 ** power
        self.vector = vector if vector else self.generate_vector(self.elements_count)
        self.coordinate_functions = list(CoordinateFunction(self.vector,
                                                            bit_number,
                                                            self.power) for bit_number in range(self.power))

    def generate_vector(self, elements_count: int) -> list[int]:
        vector = [i for i in range(elements_count)]
        random.shuffle(vector)
        return vector