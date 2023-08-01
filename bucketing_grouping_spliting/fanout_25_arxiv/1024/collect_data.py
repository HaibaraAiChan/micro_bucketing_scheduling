

# name = './nb_7_h_1024_bk_6.2.log'
# name = './nb_8_h_1024_bk_5.6.log'
# name = './nb_9_h_1024_bk_5.1.log'
# name = './nb_10_h_1024_bk_4.9.log'
# name = './nb_11_h_1024_bk_4.3.log'
# name = './nb_12_h_1024_bk_3.7.log'
# name = './nb_16_h_1024_bk_3.3.log'
name = './nb_32_h_1024_bk_2.2.log'
with open(name, 'r') as f:
    lines = f.readlines()

end_time=[]
for line in lines:
    if line.startswith("end to end time :"):
        number = float(line.split(":")[1].strip())
        # print(number)
        end_time.append(number)

print('average end to end time ')
print(sum(end_time[:3])/len(end_time[:3]))