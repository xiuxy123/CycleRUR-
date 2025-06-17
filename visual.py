import matplotlib.pyplot as plt

# # 示例数据
# parameters = [10, 20, 50, 100]  # 参数设置
# performance = [19.22, 26.05, 26.07, 33.64]  # 对应的性能表现
# #
#
# # 将参数映射到等分的位置
# num_ticks = len(parameters)
# equal_space_ticks = list(range(num_ticks))
#
# # 画折线图
# plt.plot(equal_space_ticks, performance, marker='o', linestyle='-', color='red')
#
# # 添加标题和标签
# # plt.title('BLUE vs cl_learn')
# # plt.xlabel('cl_learn')
# plt.ylabel('BLUE')
#
# plt.yticks([0, 10, 20, 30, 40])
# # plt.xticks([0, 10, 20, 50, 100])
#
# num_ticks = 4
# equal_space_ticks = list(range(num_ticks))
# tick_labels = ['10', '20', '50', '100']
#
# plt.xticks(equal_space_ticks, tick_labels)
#
# # 调整图形范围以匹配等分横坐标
# plt.xlim(-0.5, num_ticks - 0.5)
# # 显示网格线
# plt.grid(False)
#
# # 显示图例
# # plt.legend([''], loc='best', facecolor='white')
#
# # 显示图形
# plt.show()

#
# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
#
# # meteor: [10.15,]
# bleu: [2.34,2.77, 3.05, 3.14,3.15, 3.60, 4.38, 4.97, 19.01, 27.58, 29.17, 31.68,
#        30.64, 31.26, 30.76, 30.46, 30.53, 31.66, 31.65, 32.01, 30.85, 29.63, 31.42, 31.43, 32.65]  # rankloss
#
# bleu: [2.35, 2.62, 2.40, 2.60, 2.39, 24.08, 28.97, 29.70, 31.92, 32.61, 33.71, 33.58, 32.98, 34.42, 33.49, 34.51,
#        35.46, 35.79, 35.73, 34.00, 34.24, 35.38, 35.66, 34.81, 35.44]


import matplotlib.pyplot as plt

# x轴数据
x = list(range(1, 26))  # 从1到25

# y轴数据
bleu_rankloss = [2.34, 2.77, 3.05, 3.14, 3.15, 3.60, 4.38, 4.97, 19.01, 27.58, 29.17, 31.68,
                 30.64, 31.26, 30.76, 30.46, 30.53, 31.66, 31.65, 32.01, 30.85, 29.63, 31.42, 31.43, 32.65]

bleu_other = [2.35, 2.62, 2.40, 2.60, 2.39, 24.08, 28.97, 29.70, 31.92, 32.61, 33.71, 33.58,
              32.98, 34.42, 33.49, 34.51, 35.46, 35.79, 35.73, 34.00, 34.24, 35.38, 35.66, 34.81, 35.44]

bleu_src = [2.39, 3.17, 3.05, 3.17, 3.19, 3.57, 4.33, 13.80, 24.86, 24.77, 28.25, 27.52, 29.15, 30.70, 30.68,
            29.96, 29.51, 30.80, 31.93, 30.56, 31.81, 30.66, 31.29, 30.70, 31.31]

bleu_all = [2.79, 2.73, 2.57, 2.71, 19.78, 27.09, 29.31, 29.16, 31.91, 32.00, 32.35, 33.63, 32.58, 33.59,
            35.22, 32.61, 35.03, 33.38, 35.32, 34.61, 34.84, 36.39, 35.00, 34.93, 36.93]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制第三条折线
plt.plot(x, bleu_src, marker='*', linestyle='--', color='green', label='BLEU (w/o Rankloss & Retraining)')

# 绘制第二条折线
plt.plot(x, bleu_other, marker='s', linestyle='--', color='blue', label='BLEU (w/o Rankloss)')

# 绘制第一条折线
plt.plot(x, bleu_rankloss, marker='o', linestyle='--', color='yellow', label='BLEU (w/o Retraining)')

# 绘制第三条折线
plt.plot(x, bleu_all, marker='.', linestyle='--', color='red', label='BLEU (CycleRUR)')

# 添加标题和标签
plt.title('BLEU Score Comparison')
plt.xlabel('Epoch')
plt.ylabel('BLEU Score')

# 设置刻度
plt.xticks(range(0, 26, 5))  # 横坐标每5个一标记
plt.yticks(range(0, 40, 5))  # 纵坐标每5个一标记

# 添加图例
plt.legend()

# 显示图表
plt.show()



# # 示例数据
# parameters = [0.2, 0.4, 0.6, 0.8]  # 参数设置
# # bleu = [35.32, 35.77, 36.05, 35.18, 37.95]  # 对应的性能表现
# # meteor = [56.92, 58.17, 56.80, 59.27, 56.05]  # 对应的性能表现
#
# bleu = [35.77, 36.05, 35.18, 37.95]  # 对应的性能表现
# meteor = [58.17, 56.80, 59.27, 56.05]  # 对应的性能表现
# #
#
# # 将参数映射到等分的位置
# num_ticks = len(parameters)
# equal_space_ticks = list(range(num_ticks))
#
# # 画折线图
# plt.plot(equal_space_ticks, bleu, marker='o', linestyle='-', color='red', label='BLEU')
# plt.plot(equal_space_ticks, meteor, marker='.', linestyle='-', color='blue', label='METEOR')
#
# # 添加标题和标签
# # plt.title('BLUE vs cl_learn')
# plt.xlabel('loss_α')
# # plt.ylabel('BLUE')
#
# # plt.yticks([0, 10, 20, 30, 40, 50, 60])
# # plt.xticks([0, 10, 20, 50, 100])
#
# num_ticks = 4
# equal_space_ticks = list(range(num_ticks))
# tick_labels = ['0.2', '0.4', '0.6', '0.8']
#
# plt.xticks(equal_space_ticks, tick_labels)
#
# # 调整图形范围以匹配等分横坐标
# plt.xlim(-0.5, num_ticks - 0.5)
# # 显示网格线
# plt.grid(False)
#
# # 显示图例
# plt.legend()
#
# # 显示图形
# plt.show()