import matplotlib.pyplot as plt
import networkx as nx

# 创建有向图
G = nx.DiGraph()

# 添加节点
G.add_node("Data")
G.add_node("Text")
G.add_node("Data-to-Text Model")
G.add_node("Text-to-Data Model")

# 添加边
G.add_edge("Data", "Data-to-Text Model")
G.add_edge("Data-to-Text Model", "Text")
G.add_edge("Text", "Text-to-Data Model")
G.add_edge("Text-to-Data Model", "Data")

# 设置节点位置
pos = {
    "Data": (0, 1),
    "Data-to-Text Model": (1, 1),
    "Text": (2, 1),
    "Text-to-Data Model": (1, 0)
}

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

# 绘制边
nx.draw_networkx_edges(G, pos, edgelist=[("Data", "Data-to-Text Model")], arrowstyle='->', arrowsize=20)
nx.draw_networkx_edges(G, pos, edgelist=[("Data-to-Text Model", "Text")], arrowstyle='->', arrowsize=20)
nx.draw_networkx_edges(G, pos, edgelist=[("Text", "Text-to-Data Model")], arrowstyle='->', arrowsize=20)
nx.draw_networkx_edges(G, pos, edgelist=[("Text-to-Data Model", "Data")], arrowstyle='->', arrowsize=20)

# 显示图像
plt.title("Training Architecture: Data-to-Text and Text-to-Data Models")
plt.axis('off')
plt.show()
