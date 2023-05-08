from trainmodel import run_training
from fentokenizer import train_tokenizer
from perplexity import calc_perplexity
from generate_games import generate_games, generate_games_rand, generate_games_file

if __name__ == '__main__':

    model = "gpt2"
    startpositions = ['e4 e5', 'e4 c5', 'e4 d5', 'd4 d5', 'c4 e5']
    topp = 0.92
    maxlen = 1024

    # pretrain tokenizer
    train_tokenizer('data/test.fen', 'gpt2')

    # train model
    run_training('gpt2', 'data/train.fen ', 'data/test.fen')

    # testing loop
    #for i in range(1, 33):
        #checkpoint_nr = str(500 * i)

        # From common positions
        #common_avg = generate_games(model+checkpoint_nr, startpositions, topp, maxlen, 2)
        # From random positions:
        #rand_avg = generate_games_rand(model+checkpoint_nr, topp, maxlen, 4, 11)
        # From data positions:
        #data_avg = generate_games_file(model+checkpoint_nr, topp, maxlen, 2, 11, 'data/heldout.fen', 2)

        #result_string = "(" + str(checkpoint_nr) + "," + str(common_avg[1]-2) + "," + str(rand_avg[1]-10) + "," + str(data_avg[1]-10) + ")\n"
        #with open("data/results/model_name.txt", "a") as f:
            #f.write(result_string)

        # Perplexity
        #score = calc_perplexity(model+checkpoint_nr, 'data/heldout.fen')
        #with open("data/results/perplexity_model_name.txt", "a") as f:
            #f.write(str(checkpoint_nr) + " " + str(score) + "\n")