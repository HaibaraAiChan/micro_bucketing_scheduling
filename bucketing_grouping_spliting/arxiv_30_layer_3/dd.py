
data_list = [{1: 14356, 2: 6520, 3: 6082, 4: 5210, 5: 4649, 6: 4028, 7: 3423, 8: 2948, 9: 2604, 10: 26977},{1: 13428, 2: 1514, 3: 1424, 4: 1036, 5: 890, 6: 712, 7: 552, 8: 442, 9: 392, 10: 313, 11: 273, 12: 223, 13: 224, 14: 174, 15: 178, 16: 131, 17: 138, 18: 131, 19: 113, 20: 106, 21: 106, 22: 64, 23: 59, 24: 83, 25: 1369},{1: 13428}]
total = 0
for data in data_list:
    for key, value in data.items():
        total += key * value * 128 * 18 * 4 / 1024 / 1024 / 1024

print(total)
