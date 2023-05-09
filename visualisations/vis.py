import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# list of files
files = ["1milgpt2.txt", "halfmilgpt2.txt", "1milpgn.txt", "halfmilpgn.txt"]
perplexity = ["perp_1milgpt2.txt", "perp_halfmilgpt2.txt", "perp_1milpgn.txt", "perp_halfmilpgn.txt"]
models = ["large FEN", "small FEN", "large PGN", "small PGN"]
colours = ["orange", "red", "green", "cyan"]
scores = []

for file in perplexity:
    with open(file, 'r') as f:
        model_scores = ([], [], [])

        # read line by line
        for line in f:
            model_scores[0].append(float(line.strip().split()[1])) # For perplexity
            
            # For chess-specific scores
            #data = tuple(float(el) for el in line.strip('()\n').split(','))
            #model_scores[0].append(data[1])
            #model_scores[1].append(data[2])
            #model_scores[2].append(data[3])
            
        # file data to general scores
        scores.append(model_scores)
        print(scores)
            

# for each model, you have a tuple of scores for different checkpoints
large_FEN_scores = scores[0]
small_FEN_scores = scores[1]
large_PGN_scores = scores[2]
small_PGN_scores = scores[3]

# create lists of scores for all models for a test condition
all_scores = [large_FEN_scores[0], small_FEN_scores[0], large_PGN_scores[0], small_PGN_scores[0]]

# define the logarithmic regression function
def log_regression(x, a, b, c):
    return a * np.log(x + b) + c


for i, scores in enumerate(all_scores):
    x_values = [500 * n for n in range(1, len(scores) + 1)]
    
    # plot the scatter plot of all scores
    plt.scatter(x_values, scores, label=models[i], c=colours[i], s=10)
    
    # calculate the logarithmic regression line for the model
    x_data = np.array(x_values)
    y_data = np.array(scores)
    popt, pcov = curve_fit(log_regression, x_data, y_data, maxfev=20000)
    x_fit = np.linspace([500 * n for n in range(1, len(large_PGN_scores[0]) + 1)], x_values[-1], 1000)
    y_fit = log_regression(x_fit, *popt)
    
    # plot the regression line
    plt.plot(x_fit, y_fit, c=colours[i])

# add labels and legend
#plt.title("Average number of Correct Moves From the Tenth Position of Played Games")
plt.xlabel("Training Steps")
plt.ylabel("Perplexity")
plt.legend()

# show the plot
plt.show()