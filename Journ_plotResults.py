import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from matplotlib.patches import FancyBboxPatch, Rectangle


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'CSA-DRRA', 'FFO-DRRA', 'NGO-DRRA', 'HOA-DRRA', 'RFPHO-DRRA']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    fig.canvas.manager.set_window_title('Convergence Curve')
    length = np.arange(Fitness.shape[1])
    plt.plot(length, Fitness[0, :], color='r', linewidth=5, marker='*', markerfacecolor='red',
             markersize=12, label='CSA-DRRA')
    plt.plot(length, Fitness[1, :], color='g', linewidth=5, marker='*', markerfacecolor='green',
             markersize=12, label='FFO-DRRA')
    plt.plot(length, Fitness[2, :], color='b', linewidth=5, marker='*', markerfacecolor='blue',
             markersize=12, label='NGO-DRRA')
    plt.plot(length, Fitness[3, :], color='m', linewidth=5, marker='*', markerfacecolor='magenta',
             markersize=12, label='HOA-DRRA')
    plt.plot(length, Fitness[4, :], color='k', linewidth=5, marker='*', markerfacecolor='black',
             markersize=12, label='RFPHO-DRRA')
    plt.xlabel('No. of Iteration', fontname="Arial", fontsize=16, fontweight='bold', color='k')
    plt.ylabel('Cost Function', fontname="Arial", fontsize=16, fontweight='bold', color='k')
    plt.legend(loc=1)
    plt.savefig("./Results/Conv.png")
    plt.show()


def plot_Alg_Results():
    Eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['SSIM', 'PSNR', 'MI', 'ESO', 'Entropy', 'VIF', 'FMI', 'ENT', 'BRI', 'FQI', 'SF', 'CC', 'SD']
    Algorithm = ['CSA-DRRA', 'FFO-DRRA', 'NGO-DRRA', 'HOA-DRRA', 'RFPHO-DRRA']
    colors = ['#a663cc', '#f77f00', '#6a994e', '#ff6392', 'k']
    Configurations = [1, 2, 3, 4, 5]
    for b in range(len(Terms)):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
        fig.canvas.manager.set_window_title(Terms[b] + '- Image Algorithm Comparison')
        X = np.arange(len(Configurations))
        # Plotting the bars
        bar1 = ax.bar(X + 0.00, Eval[:, 0, b], color=colors[0], width=0.05, label=Algorithm[0])
        bar2 = ax.bar(X + 0.15, Eval[:, 1, b], color=colors[1], width=0.05, label=Algorithm[1])
        bar3 = ax.bar(X + 0.30, Eval[:, 2, b], color=colors[2], width=0.05, label=Algorithm[2])
        bar4 = ax.bar(X + 0.45, Eval[:, 3, b], color=colors[3], width=0.05, label=Algorithm[3])
        bar5 = ax.bar(X + 0.60, Eval[:, 4, b], color=colors[4], width=0.05, label=Algorithm[4])

        # Remove axes outline
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)
        # Custom Legend with Dot Markers, positioned at the top
        dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                       in colors]
        plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.08), fontsize=14,
                   frameon=False, ncol=len(Algorithm))
        # Loop through the bars
        for bars in [bar1, bar2, bar3, bar4, bar5]:
            for i, bar in enumerate(bars):
                # Check if the height of the bar is greater than a certain factor to modify the bar
                bar_rounding_factor = 0.08  # Customize the factor as needed
                max_col_height = max(bar.get_height() for bar in bars)  # Get the max height in this group
                if bar.get_height() > bar_rounding_factor * max_col_height:
                    # Add the rounded top bar (using FancyBboxPatch)
                    round_top = FancyBboxPatch(
                        xy=bar.get_xy(),  # Use the original bar's values to fill the new bar
                        width=bar.get_width(),
                        height=bar.get_height(),
                        color=bar.get_facecolor(),
                        boxstyle=f"round,pad=0.05,rounding_size={bar_rounding_factor}",
                        transform=ax.transData,
                        mutation_scale=1.1,
                        mutation_aspect=Eval[0, 4, b] - (
                                Eval[0, 4, b] / 2 - Eval[0, 4, b] / 4 - Eval[0, 4, b] / 16),
                    )

                    # Add a rectangular bottom bar
                    square_bottom = Rectangle(
                        xy=(bar.get_x(), bar.get_y()),
                        width=bar.get_width(),
                        height=bar.get_height() / 2,
                        color=bar.get_facecolor(),
                        transform=ax.transData
                    )

                    # Remove the original bar and add the modified ones
                    bar.remove()
                    ax.add_patch(round_top)
                    ax.add_patch(square_bottom)
        plt.xticks(X + 0.30, ('Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'), fontname="Arial", fontsize=15,
                   fontweight='bold', color='k')
        plt.ylabel(Terms[b], fontname="Arial", fontsize=16, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=16, fontweight='bold', color='#35530a')
        plt.ylim(0)
        path = "./Results/%s_Alg_bar.png" % (Terms[b])
        plt.savefig(path)
        plt.show()


def plot_Bar_Results():
    Eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['SSIM', 'PSNR', 'MI', 'ESO', 'Entropy', 'VIF', 'FMI', 'ENT', 'BRI', 'FQI', 'SF', 'CC', 'SD']
    Classifier = ['TCGAN', 'G-CNN', 'RBFN', 'Dilated Resnet', 'RFPHO-DRRA']  # 'TCGAN', 'G-CNN', 'RBFN', 'Dilated Resnet', 'RFPHO-DRRA'
    colors = ['#0077b6', '#5f0f40', '#f6aa1c', '#4f772d', '#a44a3f']
    Configurations = [1, 2, 3, 4, 5]
    for b in range(len(Terms)):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
        fig.canvas.manager.set_window_title(Terms[b] + '- Image Method Comparison')
        X = np.arange(len(Configurations))
        # Plotting the bars
        bar1 = ax.bar(X + 0.00, Eval[:, 5, b], color=colors[0], width=0.05, label=Classifier[0])
        bar2 = ax.bar(X + 0.15, Eval[:, 6, b], color=colors[1], width=0.05, label=Classifier[1])
        bar3 = ax.bar(X + 0.30, Eval[:, 7, b], color=colors[2], width=0.05, label=Classifier[2])
        bar4 = ax.bar(X + 0.45, Eval[:, 8, b], color=colors[3], width=0.05, label=Classifier[3])
        bar5 = ax.bar(X + 0.60, Eval[:, 4, b], color=colors[4], width=0.05, label=Classifier[4])

        # Remove axes outline
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)
        # Custom Legend with Dot Markers, positioned at the top
        dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                       in colors]
        plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.08), fontsize=14,
                   frameon=False, ncol=len(Classifier))
        # Loop through the bars
        for bars in [bar1, bar2, bar3, bar4, bar5]:
            for i, bar in enumerate(bars):
                # Check if the height of the bar is greater than a certain factor to modify the bar
                bar_rounding_factor = 0.08  # Customize the factor as needed
                max_col_height = max(bar.get_height() for bar in bars)  # Get the max height in this group
                if bar.get_height() > bar_rounding_factor * max_col_height:
                    # Add the rounded top bar (using FancyBboxPatch)
                    round_top = FancyBboxPatch(
                        xy=bar.get_xy(),  # Use the original bar's values to fill the new bar
                        width=bar.get_width(),
                        height=bar.get_height(),
                        color=bar.get_facecolor(),
                        boxstyle=f"round,pad=0.05,rounding_size={bar_rounding_factor}",
                        transform=ax.transData,
                        mutation_scale=1.1,
                        mutation_aspect=Eval[0, 4, b] - (
                                Eval[0, 4, b] / 4 - Eval[0, 4, b] / 16 - Eval[0, 4, b] / 32),
                    )

                    # Add a rectangular bottom bar
                    square_bottom = Rectangle(
                        xy=(bar.get_x(), bar.get_y()),
                        width=bar.get_width(),
                        height=bar.get_height() / 2,
                        color=bar.get_facecolor(),
                        transform=ax.transData
                    )

                    # Remove the original bar and add the modified ones
                    bar.remove()
                    ax.add_patch(round_top)
                    ax.add_patch(square_bottom)
        plt.xticks(X + 0.30, ('Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'), fontname="Arial", fontsize=15,
                   fontweight='bold', color='k')
        plt.ylabel(Terms[b], fontname="Arial", fontsize=16, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=16, fontweight='bold', color='#35530a')
        plt.ylim(0)
        path = "./Results/%s_Mod_bar.png" % (Terms[b])
        plt.savefig(path)
        plt.show()


def Table():
    Eval_all = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['SSIM', 'PSNR', 'MI', 'ESO', 'Entropy', 'VIF', 'FMI', 'ENT', 'BRI', 'FQI', 'SF', 'CC', 'SD']
    Methods = ['Terms', 'TCGAN', 'G-CNN', 'RBFN', 'Dilated Resnet', 'RFPHO-DRRA']
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    value_all = Eval_all[:, :, :]
    stats = np.zeros((value_all.shape[1], 5))
    for i in range(value_all.shape[2]):
        for m in range(value_all.shape[1]):
            stats[m, 0] = np.max(value_all[:, m, i])
            stats[m, 1] = np.min(value_all[:, m, i])
            stats[m, 2] = np.mean(value_all[:, m, i])
            stats[m, 3] = np.median(value_all[:, m, i])
            stats[m, 4] = np.std(value_all[:, m, i])

        Table = PrettyTable()
        Table.add_column(Methods[0], Statistics[2::])
        Table.add_column(Methods[1], stats[0, 2::])
        Table.add_column(Methods[2], stats[1, 2::])
        Table.add_column(Methods[3], stats[2, 2::])
        Table.add_column(Methods[4], stats[3, 2::])
        Table.add_column(Methods[5], stats[4, 2::])
        print('-------------------------------------------------- ', Terms[i],
              'Comparison for Segmentation of dataset', '--------------------------------------------------')
        print(Table)


if __name__ == '__main__':
    plotConvResults()
    plot_Alg_Results()
    plot_Bar_Results()
    Table()
