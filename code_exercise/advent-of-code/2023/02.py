from aocd import get_data
import time
from memory_profiler import profile


def get_game_details(puzzle_data):

    games = puzzle_data.split("\n")
    game_sets = {}

    #game_details = {game.split(":")[0]: game.split(":")[1].split(";") for game in games}
    game_details = {game[:game.index(":")]: game[game.index(":")+1:].split(";") for game in games}

    for game_id, game_info in game_details.items():
        cube_subsets = {'red': 0, 'green': 0, 'blue': 0}
        for info in game_info:
            color_configs = [configs.strip().split() for configs in info.split(",")] 
            #print(color_configs)
            for value, color in color_configs:
                cube_subsets[color] += int(value)

        game_sets[game_id] = cube_subsets

    return game_sets



def cube_conundrum(puzzle_data):

    loaded_configurations = {"red":12, "green":13, "blue":14}

    game_sets = get_game_details(puzzle_data)
    #print(game_sets)

    possible_configurations = []
    for game_id, game_configs in game_sets.items():
        #print(game_id, game_configs)
        possible = True
        for color, value in game_configs.items():
            if value > loaded_configurations[color]:
                possible = False
                break

        if possible:
            possible_configurations.append(game_id)

    #print(possible_configurations)
    sum_possible_game_ids = sum([int(num) for i in possible_configurations for num in i if num.isdigit()])

    return sum_possible_game_ids



def part1(data):

    input_data = cube_conundrum(data)
    print(f'''Sum of valid game id's: {input_data} ''')

    return input_data 



def part2(data):

    return len(data) 


def test_part1():
    assert part1(test_data) == 8


def test_part2():
    assert part2(test_data) == 320


data = get_data(day=2, year=2023)

print(part1(data))
print(part2(data))

test_data = """Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green
Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue
Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red
Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red
Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green"""


print(part1(test_data))
























