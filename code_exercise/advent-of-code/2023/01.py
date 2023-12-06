from aocd import get_data
import time
from memory_profiler import profile



def search_replace_word_digits(input_data):

    valid_digit = {
        'one': 'one1one', 'two': 'two2two', 'three': 'three3three', 'four': 'four4four', 'five': 'five5five',
        'six': 'six6six', 'seven': 'seven7seven', 'eight': 'eight8eight', 'nine': 'nine9nine'
    }
    
    word_digits = [input_data]
    for word, digit in valid_digit.items():
        word_digits = [char.replace(word, digit) for char in word_digits]
    
    word_digits = word_digits[0].split("\n")
    return word_digits


def get_calibration_value(input_data):
    
    input_values = []
    for char in input_data:
        input_value = ''.join([i for i in char if i.isdigit()])
        if input_value:
            input_values.append(str(input_value))
        else:
            input_values.append(str(0))

    calibration_values = []

    for i in input_values:
        calibration_value = i[0] + i[-1] 
        calibration_values.append(calibration_value)
        continue     

    return calibration_values


@profile
def sum_calibration_values(calibration_values):

    total_value_calibration = sum(int(i) for i in calibration_values)

    return total_value_calibration


def part1(data):

    input_data = [s for s in data.split("\n")]
    calibration_values = get_calibration_value(input_data)
    total_value_calibration = sum_calibration_values(calibration_values)

    return total_value_calibration


def part2(data):

    input_data =  search_replace_word_digits(data)
    calibration_values = get_calibration_value(input_data)
    total_value_calibration = sum_calibration_values(calibration_values)

    return total_value_calibration


def test_part1():
    assert part1(test_data_1) == 142


def test_part2():
    assert part2(test_data_2) == 281


data = get_data(day=1, year=2023)

print(part1(data))
print(part2(data))

test_data_1 = """1abc2
pqr3stu8vwx
a1b2c3d4e5f
treb7uchet"""


test_data_2 = """two1nine
    eightwothree
    abcone2threexyz
    xtwone3four
    4nineeightseven2
    zoneight234
    7pqrstsixteen"""







#if __name__=="__main__":

    # test_data_1 = """ag3jj56
    #             sjkdf34sdf
    #             a1b2c3d4e5f

    #             /$§+ä
    #             erzhhdui 
    #             dffj8CVc"""
    
    # test_data_2 = """two1nine
    #     five57seven7dfgninine
    #     eightwothree
    #     abcone2threexyz
    #     xtwone3four
    #     4nineeightseven2
    #     zoneight234
    #     7pqrstsixteen"""

    # start_time = time.time()

    # # part 1
    # input_data_1 = [s for s in test_data_1.split("\n")]
    # calibration_values_1 = get_calibration_value(input_data_1)
    # total_value_calibration_1 = sum_calibration_values(calibration_values_1)
    # print(total_value_calibration_1)

    # # part 2
    # input_data_2 =  search_replace_word_digits(test_data_2)
    # calibration_values_2 = get_calibration_value(input_data_2)
    # total_value_calibration_2 = sum_calibration_values(calibration_values_2)
    # print(total_value_calibration_2)

    # end_time = time.time()
    # total_time = end_time - start_time
    # print(f"Total time (s): {total_time}")
















