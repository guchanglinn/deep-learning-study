import matplotlib.pyplot as plt

x , y = [] ,[]
with open('../dataset/food_truck_data.txt') as A:
    for eachline in A:
        s = eachline.split(',')
        x.append(float(s[0]))
        y.append(float(s[1]))
for i in range(5):
    print(x[i],y[i])



_, ax = plt.subplots()
ax.scatter(x, y, marker="x", c="red")
plt.title("Food Truck Dataset", fontsize=16)
plt.xlabel("City Population in 10,000s", fontsize=14)
plt.ylabel("Food Truck Profit in 10,000s", fontsize=14)
plt.axis([4, 25, -5, 25])
plt.show()