#For attention model
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.arange(10)
# y_1 = [13.7516, 4.7995, 3.8258, 3.6387, 3.6166, 3.6146, 3.6139, 3.6147, 3.6138, 3.6140, 3.6145, 3.6146, 3.6145, 3.6147,
#        3.6148]
# x_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# y_2 = [25.6733, 25.6223, 23.9142, 24.2588, 24.2360, 24.2173, 24.2144, 24.2145, 24.2146, 24.2146, 24.2146, 24.2146,
#        24.2146, 24.2146, 24.2146]
# x_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#
# plt.plot(x_1, y_1)
# plt.plot(x_2, y_2)
# plt.legend(['Train', 'Test'], loc='upper right')
# plt.xlabel('Number of epochs')
# plt.ylabel('Accuracy')
# plt.show()


#For pointer network model
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.arange(10)
# y_1 = [14.6797, 4.6039, 3.6527, 3.4747, 3.4549, 3.4530, 3.4525, 3.4518, 3.4520, 3.4524, 3.4533, 3.4529, 3.4523, 3.4521,
#        3.4524]
# x_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# y_2 = [26.6837, 28.9170, 27.2997, 27.3443, 27.4207, 27.4252, 27.4257, 27.4257, 27.4257, 27.4257, 27.4257, 27.4257,
#        27.4257, 27.4257, 27.4257]
# x_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#
# plt.plot(x_1, y_1)
# plt.plot(x_2, y_2)
# plt.legend(['Train', 'Test'], loc='upper right')
# plt.xlabel('Number of epochs')
# plt.ylabel('Perplexity')
# plt.show()

#For accuracy
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.arange(10)
# y_1 = [0.5558,0.5663,0.5792,0.5789,0.5790,0.5791,0.5791,0.5791,0.5791,0.5791,0.5791,0.5791,0.5791,0.5791,0.5791]
# x_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# y_2 = [0.5602,0.5647,0.5781,0.5802,0.5798,0.5798,0.5798,0.5798,0.5798,0.5798,0.5798,0.5798,0.5798,0.5798,0.5798]
# x_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#
# plt.plot(x_1, y_1)
# plt.plot(x_2, y_2)
# plt.legend(['Seq2Seq Model with attention', 'Seq2Seq Model with pointer n/w'], loc='upper right')
# plt.xlabel('Number of epochs')
# plt.ylabel('Accuracy')
# plt.show()



# #For sentiment classifier data
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.arange(10)
# y_1 = [85.94,  88.52, 89.15, 88.77, 89.23, 89.03, 88.86, 89.01, 87.79, 88.32]
# x_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#
# y_2 = [74.26,  87.01, 91.33, 94.77, 97.09, 98.58, 99.23, 99.57, 99.80, 99.77]
# x_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#
# plt.plot(x_1, y_1)
# plt.plot(x_2, y_2)
# plt.legend(['Validation set', 'Training set'], loc='upper right')
# plt.xlabel('Number of epochs')
# plt.ylabel('Accuracy')
# plt.show()


import matplotlib.pyplot as plt

# x-coordinates of left sides of bars
left = [1, 2, 3]

# heights of bars
height = [53.384, 88.66, 53.1]

# labels for bars
tick_label = ['Attention Mechanism', 'Original IMDB Test dataset', 'Pointer Network']

# plotting a bar chart
plt.bar(left, height, tick_label=tick_label,
        width=0.8, color=['red', 'green','blue'])

# naming the x-axis
plt.xlabel('x - axis')
# naming the y-axis
plt.ylabel('y - axis')
# plot title
plt.title('Sentence classification Test Accuracy')

# function to show the plot
plt.show()

