from transformers import GPT2TokenizerFast, AutoModelForCausalLM
import pandas as pd
import io
import chess.pgn
from datetime import date
import statistics
import random
from random import randrange

# game generation from Andreas Stöckl https://github.com/astoeckl/ChessTransformer/blob/master/generate_games.py

def savepgn(df,model_type,topp,maxlen,eval_type):
    # Korrekten Teil als PGN speichern
    for i in range(df.shape[0]):
        gamestr = str(df.iloc[i][0])
        pgn = io.StringIO(gamestr)
        game = chess.pgn.read_game(pgn)
        game.headers["Event"] = "Generated moves"
        game.headers["Site"] = "Tartu"
        game.headers["Date"] = str(date.today())
        game.headers["Round"] = str(i)
        game.headers["White"] = str(model_type)
        game.headers["Black"] = str(model_type)
        filenamepgn = 'data/games/pgn_partien_'+ eval_type + model_type + str(int(topp*100)) + str(maxlen) + '.pgn'
        print(game, file=open(filenamepgn, "a"), end="\n\n")


def count_moves(df):
    erglist = []

    for san in df['Zuege']:
        partie = san.split()
        board = chess.Board()

        i = 0
        for zug in partie:
            try:
                board.push_san(zug)
                i = i + 1
            except:
                erglist.append(i)
                break
    return (erglist,statistics.mean(erglist))


def generate_games(model_type, startpositionen, topp, maxlen, games_per_position):
    tokenizer = GPT2TokenizerFast.from_pretrained("data/pgntokenizer")
    model = AutoModelForCausalLM.from_pretrained("data/pgnmodels_" + model_type)

    outputlist = []
    for start in startpositionen:
        input_ids = tokenizer.encode(start, return_tensors='pt')

        # generate text until the output length (which includes the context length) reaches max_length
        for i in range(games_per_position):
            output = model.generate(input_ids,
                                    max_length=maxlen,
                                    top_p=topp,
                                    do_sample=True)
            zuege = tokenizer.decode(output[0], skip_special_tokens=True)
            outputlist.append(zuege)

    df = pd.DataFrame(outputlist, columns=["Zuege"])
    #filename = 'data/games/pgn_partien_'+model_type+str(int(topp*100))+str(maxlen)+'.csv'
    #df.to_csv(filename, index=False)
    #savepgn(df, model_type, topp, maxlen,"")

    # Wie viele Züge sind korrekt pro Partie / Durchschnitt
    # Von Startpositionen
    counts = count_moves(df)
    return counts


def random_move(board):
    move = random.choice(list(board.legal_moves))
    return board.san(move)


def generate_games_rand(model_type, topp, maxlen, games_per_position, depth):
    # Züege ab zufälliger Position generieren - Tiefe
    # Generate random Position
    gamestring = ""
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        move = random.choice(list(board.legal_moves))
        gamestring = gamestring + " " + board.san(move)
        board.push(move)

    gamelist = gamestring.split()
    start = ' '.join(gamelist[:depth])

    tokenizer = GPT2TokenizerFast.from_pretrained("data/pgntokenizer")
    model = AutoModelForCausalLM.from_pretrained("data/pgnmodels_" + model_type)
    outputlist = []
    input_ids = tokenizer.encode(start, return_tensors='pt')

    # generate text until the output length (which includes the context length) reaches max_length
    for i in range(games_per_position):
        output = model.generate(input_ids,
                                max_length=maxlen,
                                top_p=topp,
                                do_sample=True)
        zuege = tokenizer.decode(output[0], skip_special_tokens=True)
        outputlist.append(zuege)

    df = pd.DataFrame(outputlist, columns=["Zuege"])
    #filename = 'data/games/pgn_partien_rand_' + model_type + str(int(topp * 100)) + str(maxlen) + '.csv'
    #df.to_csv(filename, index=False)
    #savepgn(df, model_type, topp, maxlen,"rand")

    counts = count_moves(df)
    return counts

# Zug ab zufälliger Position aus den Trainingsdaten generieren
# Generate random Position from Games
def generate_games_file(model_type, topp, maxlen, games_per_position, depth, file, position_per_file):

    tokenizer = GPT2TokenizerFast.from_pretrained("data/pgntokenizer")
    model = AutoModelForCausalLM.from_pretrained("data/pgnmodels_" + model_type)

    with open(file, 'r') as f:
        lines = f.readlines()
    df = pd.DataFrame(lines)

    outputlist = []

    for i in range(position_per_file):
        # Generate random Position from Games
        gamestring = str(df.iloc[randrange(df.shape[0])][0])
        gamelist = gamestring.split()
        start = ' '.join(gamelist[:depth])

        # encode context the generation is conditioned on
        input_ids = tokenizer.encode(start, return_tensors='pt')

        # generate text until the output length (which includes the context length) reaches maxlength
        for i in range(games_per_position):
            output = model.generate(input_ids,
                                    max_length=maxlen,
                                    top_p=topp,
                                    do_sample=True)
            zuege = tokenizer.decode(output[0], skip_special_tokens=True)
            outputlist.append(zuege)

    df = pd.DataFrame(outputlist, columns=["Zuege"])
    #filename = 'data/games/pgn_file_' + model_type + str(int(topp * 100)) + str(maxlen) + '.csv'
    #df.to_csv(filename, index=False)
    #savepgn(df, model_type, topp, maxlen,"file")

    counts = count_moves(df)
    return counts