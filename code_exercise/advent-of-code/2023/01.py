from aocd import get_data



def get_calibration_value(input):
    
    input_values = []
    for char in input:
        input_value = ''.join([i for i in char if i.isdigit()])
        if input_value:
            input_values.append(str(input_value))
        else:
            input_values.append(str(0))

    print(input_values)


    calibration_values = []

    for i in input_values:
        if len(input_values)==1:
            calibration_value = i[0] + i[0] 
            calibration_values.append(calibration_value)
        else:
            calibration_value = i[0] + i[-1] 
            calibration_values.append(calibration_value)

        continue     

    return calibration_values

def sum_calibration_values(calibration_values):

    total_value_calibration = sum(int(i) for i in calibration_values)

    return total_value_calibration



def part1(data):
    input = [s for s in data.split("\n")]

    calibration_values = get_calibration_value(input)
    total_value_calibration = sum_calibration_values(calibration_values)


    return total_value_calibration


def part2(data):
    input = [s for s in data.split("\n")]

    return len(input)


def test_part1():
    assert part1(test_data) == 150


def test_part2():
    assert part2(test_data) == 900


data = get_data(day=1, year=2023)

print(part1(data))
print(part2(data))

test_data = """1abc2
pqr3stu8vwx
a1b2c3d4e5f
treb7uchet"""





if __name__=="__main__":

    test_data_1 = """ag3jj56
                sjkdf34sdf
                a1b2c3d4e5f

                /$§+ä
                erzhhdui 
                dffj8CVc"""
    

    input = [s for s in test_data_1.split("\n")]

    calibration_values = get_calibration_value(input)
    print(calibration_values)

    total_value_calibration = sum_calibration_values(calibration_values)

    print(total_value_calibration)




