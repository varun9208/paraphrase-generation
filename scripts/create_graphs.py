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


# import matplotlib.pyplot as plt
#
# # x-coordinates of left sides of bars
# left = [1, 2, 3]
#
# # heights of bars
# height = [53.384, 88.66, 53.1]
#
# # labels for bars
# tick_label = ['Attention Mechanism', 'Original IMDB Test dataset', 'Pointer Network']
#
# # plotting a bar chart
# plt.bar(left, height, tick_label=tick_label,
#         width=0.8, color=['red', 'green','blue'])
#
# # naming the x-axis
# plt.xlabel('x - axis')
# # naming the y-axis
# plt.ylabel('y - axis')
# # plot title
# plt.title('Sentence classification Test Accuracy')
#
# # function to show the plot
# plt.show()


#For ROC curve
import matplotlib.pyplot as plt
import numpy as np
#
# x = np.arange(10)
#
x_0 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y_0 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# y_first = [1.0, 1.0, 1.0, 1.0, 0.97608, 0.64216, 0.00888, 0.0, 0.0, 0.0]
# x_first = [1.0, 1.0, 1.0, 1.0, 0.98432, 0.67328, 0.011759999999999993, 0.0, 0.0, 0.0]
# y_second = [1.0, 0.99744, 0.9756, 0.94216, 0.90256, 0.856, 0.78904, 0.68056, 0.4516, 0.10744, 0.0]
# x_second = [1.0, 0.95208, 0.81208, 0.66952, 0.55392, 0.44920000000000004, 0.3496, 0.24248000000000003, 0.10960000000000003, 0.012959999999999972, 0.0]
# y_third = [1.0, 0.9992, 0.9932, 0.968, 0.8892, 0.69928, 0.4004, 0.10608, 0.00688, 8e-05, 0.0]
# x_third = [1.0, 0.99944, 0.99256, 0.94352, 0.7804, 0.47504, 0.16208, 0.022719999999999962, 0.00039999999999995595, 0.0, 0.0]
# y_fourth=[1.0, 0.946, 0.91296, 0.87648, 0.84184, 0.80824, 0.7636, 0.7084, 0.63888, 0.5144, 0.0]
# x_fourth=[1.0, 0.44752000000000003, 0.33984000000000003, 0.27256, 0.22263999999999995, 0.18208000000000002, 0.14224000000000003, 0.11287999999999998, 0.07896000000000003, 0.04264000000000001, 0.0]
#
# plt.plot(x_first, y_first)
# plt.plot(x_second, y_second)
# plt.plot(x_third, y_third)
# plt.plot(x_fourth, y_fourth)
# plt.plot(x_0, y_0, '--')
# plt.legend(['RNN', 'BIRNN', 'FastText', 'CNN', 'Random Guess'], loc='bottom right')
# plt.xlabel('False Positive Ratio')
# plt.ylabel('True Positive Ratio')
# plt.show()


y_attn=[1.0, 0.68128, 0.4068, 0.24872, 0.15104, 0.09392, 0.05528, 0.0324, 0.01528, 0.00512, 0.0]
x_attn =[1.0, 0.42791999999999997, 0.19311999999999996, 0.09511999999999998, 0.04815999999999998, 0.02632000000000001, 0.010800000000000032, 0.005680000000000018, 0.0026399999999999757, 0.00048000000000003595, 0.0]


y_ptr=[1.0, 0.76456, 0.47976, 0.28696, 0.16664, 0.0996, 0.05648, 0.03008, 0.01088, 0.00256, 0.0]
x_ptr=[1.0, 0.5731999999999999, 0.29335999999999995, 0.14600000000000002, 0.0716, 0.035760000000000014, 0.016959999999999975, 0.008560000000000012, 0.0025600000000000067, 0.001040000000000041, 0.0]

y_fourth=[1.0, 0.946, 0.91296, 0.87648, 0.84184, 0.80824, 0.7636, 0.7084, 0.63888, 0.5144, 0.0]
x_fourth=[1.0, 0.44752000000000003, 0.33984000000000003, 0.27256, 0.22263999999999995, 0.18208000000000002, 0.14224000000000003, 0.11287999999999998, 0.07896000000000003, 0.04264000000000001, 0.0]

plt.plot(x_attn, y_attn)
plt.plot(x_ptr, y_ptr)
plt.plot(x_fourth, y_fourth)
plt.plot(x_0, y_0, '--')
plt.legend(['Sentences generated using Attention', 'Sentences generated using Pointer', 'Original Test Samples', 'Random Guess'], loc='bottom right')
plt.xlabel('False Positive Ratio')
plt.ylabel('True Positive Ratio')
plt.show()
