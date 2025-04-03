# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import torch



# FIGSIZE = (6, 4)
# LINEWIDTH = 2.0
# FONTSIZE = 12
# def plot_loss_accs(
#     statistics, multiple_runs=False, log_x=False, log_y=False, 
#     figsize=FIGSIZE, linewidth=LINEWIDTH, fontsize=FONTSIZE,
#     fileName=None, filePath=None, show=True
#     ):

#     rows, cols = 1, 2
#     fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))
#     color_1 = 'tab:blue' # #1f77b4
#     color_2 = 'tab:red' # #d62728
    
#     same_steps = False
#     if multiple_runs :

#         all_steps = statistics["all_steps"]
#         same_steps = all(len(steps) == len(all_steps[0]) for steps in all_steps) # Check if all runs have the same number of steps
#         if same_steps :
#             all_steps = np.array(all_steps[0]) + 1e-0 # Add 1e-0 to avoid log(0)
#         else :
#             all_steps = [np.array(steps) + 1e-0 for steps in all_steps] # Add 1e-0 to avoid log(0)
#             color_indices = np.linspace(0, 1, len(all_steps))
#             colors = plt.cm.viridis(color_indices)
#     else :
#         all_steps = np.array(statistics["all_steps"]) + 1e-0

#     for i, key in enumerate(["accuracy", "loss"]) :
#         ax = fig.add_subplot(rows, cols, i+1)
#         if multiple_runs :
#             # zs = np.array(statistics["train"][key])

#             print(f"Clé: {key}, Type: {type(statistics['train'][key])}")
#             if isinstance(statistics["train"][key], list):
#                 print(f"Exemple d'élément: {type(statistics['train'][key][0])}")



#             # zs = np.array([t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in statistics["train"][key]])
# # Vérifiez si le tenseur est sur le GPU, puis déplacez-le vers le CPU avant de le convertir en Numpy
#             zs = np.array([t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t for t in statistics["train"][key]])

#             if same_steps :
#                 zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
#                 #ax.errorbar(all_steps, zs_mean, yerr=zs_std, fmt=f'-', color=color_1, label=f"Train", lw=linewidth)
#                 ax.plot(all_steps, zs_mean, '-', color=color_1, label=f"Train", lw=linewidth)
#                 ax.fill_between(all_steps, zs_mean-zs_std, zs_mean+zs_std, color=color_1, alpha=0.2)
#             else :  
#                 for j, z in enumerate(zs) :
#                     ax.plot(all_steps[j], z, '-', color=colors[j], label=f"Train", lw=linewidth/2)

#             zs = np.array(statistics["test"][key])
#             if same_steps :
#                 zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
#                 ax.plot(all_steps, zs_mean, '-', color=color_2, label=f"Eval", lw=linewidth)
#                 ax.fill_between(all_steps, zs_mean-zs_std, zs_mean+zs_std, color=color_2, alpha=0.2)
#             else :  
#                 for j, z in enumerate(zs) :
#                     ax.plot(all_steps[j], z, '--', color=colors[j], label=f"Eval", lw=linewidth/2)

#         else :
#               # Convertir les données de train et de test en numpy, si ce sont des tenseurs CUDA
#             # train_data = statistics["train"][key]
#             # if torch.is_tensor(train_data):
#             #     train_data = train_data.cpu().numpy()
#             # test_data = statistics["test"][key]
#             # if torch.is_tensor(test_data):
#             #     test_data = test_data.cpu().numpy()

            
#             print(type(train_data))
#             print(train_data.device)

#             # Convertir les données de train et de test en numpy, si ce sont des tenseurs CUDA
#             train_data = statistics["train"][key]
#             if isinstance(train_data, torch.Tensor):
#                 train_data = train_data.detach().cpu().numpy()
#             elif isinstance(train_data, list):
#                 train_data = np.array([t.detach().cpu().item() if isinstance(t, torch.Tensor) else t for t in train_data])

#             test_data = statistics["test"][key]
#             if isinstance(test_data, torch.Tensor):
#                 test_data = test_data.detach().cpu().numpy()
#             elif isinstance(test_data, list):
#                 test_data = np.array([t.detach().cpu().item() if isinstance(t, torch.Tensor) else t for t in test_data])


#             ax.plot(all_steps, train_data, "-", color=color_1, label=f"Train", lw=linewidth) 
#             ax.plot(all_steps, test_data, "-", color=color_2, label=f"Eval", lw=linewidth)


#             # ax.plot(all_steps, statistics["train"][key], "-", color=color_1,  label=f"Train", lw=linewidth) 
#             # ax.plot(all_steps, statistics["test"][key], "-", color=color_2,  label=f"Eval", lw=linewidth) 

#             # ax.plot(all_steps, train_data, "-", color=color_1, label=f"Train", lw=linewidth) 
#             # ax.plot(all_steps, test_data, "-", color=color_2, label=f"Eval", lw=linewidth)



#         if log_x : ax.set_xscale('log')
#         #if log_y : ax.set_yscale('log')
#         if log_y and key=="loss" : ax.set_yscale('log') # No need to log accuracy
#         ax.tick_params(axis='y', labelsize='x-large')
#         ax.tick_params(axis='x', labelsize='x-large')
#         ax.set_xlabel("Training Steps (t)", fontsize=fontsize)
#         if key=="accuracy": s = "Accuracy"
#         if key=="loss": s = "Loss"
#         #ax.set_ylabel(s, fontsize=fontsize)
#         ax.set_title(s, fontsize=fontsize)
#         ax.grid(True)
#         if multiple_runs and (not same_steps) :
#             legend_elements = [Line2D([0], [0], color='k', lw=linewidth, linestyle='-', label='Train'),
#                             Line2D([0], [0], color='k', lw=linewidth, linestyle='--', label='Eval')]
#             ax.legend(handles=legend_elements, fontsize=fontsize)
#         else :
#             ax.legend(fontsize=fontsize)

#     if fileName is not None and filePath is not None :
#         os.makedirs(filePath, exist_ok=True)
#         plt.savefig(f"{filePath}/{fileName}"  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

#     if show : plt.show()
#     else : plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

FIGSIZE = (6, 4)
LINEWIDTH = 2.0
FONTSIZE = 12

def convert_to_cpu(tensor_or_list):
    """
    Cette fonction prend soit un tenseur PyTorch, soit une liste de tenseurs PyTorch et 
    les déplace vers le CPU avant de les convertir en Numpy.
    """
    if isinstance(tensor_or_list, list):
        # Si c'est une liste, applique la conversion à chaque élément de la liste
        return np.array([convert_to_cpu(t) for t in tensor_or_list])
    elif isinstance(tensor_or_list, torch.Tensor):
        # Déplacer le tenseur vers le CPU si nécessaire
        return tensor_or_list.detach().cpu().numpy()
    else:
        # Si ce n'est pas un tenseur ou une liste, on suppose qu'il est déjà prêt
        return tensor_or_list

def plot_loss_accs(
    statistics, multiple_runs=False, log_x=False, log_y=False, 
    figsize=FIGSIZE, linewidth=LINEWIDTH, fontsize=FONTSIZE,
    fileName=None, filePath=None, show=True
    ):

    rows, cols = 1, 2
    fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))
    color_1 = 'tab:blue'  # #1f77b4
    color_2 = 'tab:red'   # #d62728
    
    same_steps = False
    if multiple_runs:
        all_steps = statistics["all_steps"]
        same_steps = all(len(steps) == len(all_steps[0]) for steps in all_steps)  # Check if all runs have the same number of steps
        if same_steps:
            all_steps = np.array(all_steps[0]) + 1e-0  # Add 1e-0 to avoid log(0)
        else:
            all_steps = [np.array(steps) + 1e-0 for steps in all_steps]  # Add 1e-0 to avoid log(0)
            color_indices = np.linspace(0, 1, len(all_steps))
            colors = plt.cm.viridis(color_indices)
    else:
        all_steps = np.array(statistics["all_steps"]) + 1e-0

    for i, key in enumerate(["accuracy", "loss"]):
        ax = fig.add_subplot(rows, cols, i+1)
        if multiple_runs:
            print(f"Clé: {key}, Type: {type(statistics['train'][key])}")
            if isinstance(statistics["train"][key], list):
                print(f"Exemple d'élément: {type(statistics['train'][key][0])}")

            # Convertir en numpy après avoir déplacé vers le CPU
            zs = convert_to_cpu(statistics["train"][key])

            if same_steps:
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                ax.plot(all_steps, zs_mean, '-', color=color_1, label=f"Train", lw=linewidth)
                ax.fill_between(all_steps, zs_mean - zs_std, zs_mean + zs_std, color=color_1, alpha=0.2)
            else:
                for j, z in enumerate(zs):
                    ax.plot(all_steps[j], z, '-', color=colors[j], label=f"Train", lw=linewidth / 2)

            zs = convert_to_cpu(statistics["test"][key])
            if same_steps:
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                ax.plot(all_steps, zs_mean, '-', color=color_2, label=f"Eval", lw=linewidth)
                ax.fill_between(all_steps, zs_mean - zs_std, zs_mean + zs_std, color=color_2, alpha=0.2)
            else:
                for j, z in enumerate(zs):
                    ax.plot(all_steps[j], z, '--', color=colors[j], label=f"Eval", lw=linewidth / 2)

        else:
            # Convertir les données de train et de test en numpy après les avoir déplacées vers le CPU
            train_data = statistics["train"][key]
            train_data = convert_to_cpu(train_data)

            test_data = statistics["test"][key]
            test_data = convert_to_cpu(test_data)

            ax.plot(all_steps, train_data, "-", color=color_1, label=f"Train", lw=linewidth)
            ax.plot(all_steps, test_data, "-", color=color_2, label=f"Eval", lw=linewidth)

        if log_x:
            ax.set_xscale('log')
        if log_y and key == "loss":
            ax.set_yscale('log')  # No need to log accuracy
        ax.tick_params(axis='y', labelsize='x-large')
        ax.tick_params(axis='x', labelsize='x-large')
        ax.set_xlabel("Training Steps (t)", fontsize=fontsize)
        if key == "accuracy":
            s = "Accuracy"
        if key == "loss":
            s = "Loss"
        ax.set_title(s, fontsize=fontsize)
        ax.grid(True)
        if multiple_runs and (not same_steps):
            legend_elements = [
                Line2D([0], [0], color='k', lw=linewidth, linestyle='-', label='Train'),
                Line2D([0], [0], color='k', lw=linewidth, linestyle='--', label='Eval')
            ]
            ax.legend(handles=legend_elements, fontsize=fontsize)
        else:
            ax.legend(fontsize=fontsize)

    if fileName is not None and filePath is not None:
        os.makedirs(filePath, exist_ok=True)
        plt.savefig(f"{filePath}/{fileName}" + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

    if show:
        plt.show()
    else:
        plt.close()
