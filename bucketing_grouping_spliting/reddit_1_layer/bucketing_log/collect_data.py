


name = './nb_3_bucketing.log'
with open(name, 'r') as f:
    lines = f.readlines()

end_time=[]
for line in lines:
    if line.startswith("end to end time :"):
        number = float(line.split(":")[1].strip())
        # print(number)
        end_time.append(number)

print('average end to end time ')
print(sum(end_time[3:])/len(end_time[3:]))


