import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [50, 10]

xgb.plot_tree(xgb_clf, num_trees=0)
plt.savefig("Tree.png", dpi=100)