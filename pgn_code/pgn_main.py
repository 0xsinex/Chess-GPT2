from pgn_trainmodel import run_training
from pgntokenizer import train_tokenizer
from pgn_perplexity import calc_perplexity
from pgn_generate_games import generate_games, generate_games_rand,generate_games_file

if __name__ == '__main__':

    model = "pgnmodel/checkpoint-"
    startpositions = ['e4 e5', 'e4 c5', 'e4 d5', 'd4 d5', 'c4 e5']
    topp = 0.92
    maxlen = 500

    # pretrain tokenizer
    #train_tokenizer('data/tokenizer.pgn', 'gpt2')

    # train model
    #run_training('gpt2', 'data/train.pgn', 'data/test.pgn')

    # testing loop
    for i in range(1, 56):  # 111 steps for large and 56 for small
        checkpoint_nr = str(500 * i)
        # From common positions
        #common_avg = generate_games(model+checkpoint_nr, startpositions, topp, maxlen, 2)
        # From random positions:
        #rand_avg = generate_games_rand(model+checkpoint_nr, topp, maxlen, 4, 11)
        # From data positions:
        #data_avg = generate_games_file(model+checkpoint_nr, topp, maxlen, 2, 11, 'data/heldout.pgn', 2)
        #result_string = "(" + str(checkpoint_nr) + "," + str(common_avg[1] - 2) + "," + str(rand_avg[1] - 10) + "," + str(data_avg[1] - 10) +")\n"
        #with open("data/results/pgnmodel.txt", "a") as f:
            #f.write(result_string)
        #print(result_string)

        # Perplexity
        score = calc_perplexity(model+checkpoint_nr, 'data/heldout.pgn')
        with open("data/results/perp_model_name.txt", "a") as f:
            f.write(str(checkpoint_nr) + " " + str(score) + "\n")
