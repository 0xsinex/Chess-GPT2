from transformers import GPT2TokenizerFast, AutoModelForCausalLM
import pandas as pd
import io
import chess
from datetime import date
import statistics
import random
from random import randrange

# game generation based on Andreas Stöckl https://github.com/astoeckl/ChessTransformer/blob/master/generate_games.py

def savefen(df,model_type,topp,maxlen,eval_type):
    # Save correct part as FEN
    for i in range(df.shape[0]):
        gamestr = str(df.iloc[i][0])
        filenamefen = 'data/games/fengame_' + eval_type + model_type + str(int(topp*100)) + str(maxlen) + '.txt'
        print(gamestr, file=open(filenamefen, "a"), end="\n")


def count_moves(df):
    # take board state
    # try all legal moves and put their fen states in a list
    # if made fen move is in the list
    # else:
    # add the movecount to the counts list
    # break and calculate statistics
    counts = []

    for fen in df['Move']:
        game = fen.split(';')[1:]
        board = chess.Board()
        board.reset()
        i = 0
        for mv in game:
            fen_move = mv[:mv.rfind(" ")] # remove en passant, which can be faulty in the training data
            if fen_move in legal_moves(board):
                board.set_fen(fen_move)
                i = i + 1
            else:
                counts.append(i)
                break
    return counts, statistics.mean(counts)

def legal_moves(board_state):
    # generates fen states of all legal moves
    fens = []
    all_moves = board_state.legal_moves

    for move in all_moves:
        board_state.push(move)
        smaller_state = board_state.fen()[:-4] # removes half-move clock and full-move count
        fens.append(smaller_state[:smaller_state.rfind(" ")]) # removes en passant move, which can be faulty in the file
        board_state.pop()

    return fens


def generate_games(model_type, startposition, topp, maxlen, games_per_position):
    # generate games from a list of typical opening positions after two moves
    tokenizer = GPT2TokenizerFast.from_pretrained("data/fentokenizer")
    model = AutoModelForCausalLM.from_pretrained("data/model_"+model_type)

    outputlist = []
    for start in startposition:

        # convert startposition to FEN
        board = chess.Board()
        fen_start = board.fen()[:-4]
        for pgn_move in start.split():
            board.push(board.parse_san(pgn_move))
            fen_start = fen_start + ";" + board.fen()[:-4]

        input_ids = tokenizer.encode(fen_start, return_tensors='pt')

        # generate text until the output length (which includes the context length) reaches max_length
        for i in range(games_per_position):
            output = model.generate(input_ids,
                                    max_length=maxlen,
                                    top_p=topp,
                                    do_sample=True)
            move = tokenizer.decode(output[0], skip_special_tokens=True)
            outputlist.append(move)

    df = pd.DataFrame(outputlist, columns=["Move"])
    #filename = 'data/games/games_'+model_type+str(int(topp*100))+str(maxlen)+'.csv'
    #df.to_csv(filename, index=False)
    #savefen(df, model_type, topp, maxlen,"")

    # How many moves are correct per game / average
    # From starting positions
    counts = count_moves(df)
    return counts

def random_move(board):
    move = random.choice(list(board.legal_moves))
    return board.san(move)


def generate_games_rand(model_type, topp, maxlen, games_per_position, depth):
    # Generate moves from random position - depth
    # Generate random Position
    board = chess.Board()
    gamestring_fen = board.fen()[:-4]
    gamestring_pgn = ""
    board = chess.Board()
    # generates random moves, saves in pgn and fen
    while not board.is_game_over(claim_draw=True):
        move = random.choice(list(board.legal_moves))
        gamestring_pgn = gamestring_pgn + " " + board.san(move)
        board.push(move)
        gamestring_fen = gamestring_fen + ";" + board.fen()[:-4]

    gamelist = gamestring_fen.split(";")
    start = ';'.join(gamelist[:depth])

    tokenizer = GPT2TokenizerFast.from_pretrained("data/fentokenizer")
    model = AutoModelForCausalLM.from_pretrained("data/model_" + model_type)

    outputlist = []
    input_ids = tokenizer.encode(start, return_tensors='pt')

    # generate text until the output length (which includes the context length) reaches max_length
    for i in range(games_per_position):
        output = model.generate(input_ids,
                                max_length=maxlen,
                                top_p=topp,
                                do_sample=True)
        move = tokenizer.decode(output[0], skip_special_tokens=True)
        outputlist.append(move)

    df = pd.DataFrame(outputlist, columns=["Move"])
    #filename = 'data/games/games_rand_' + model_type + str(int(topp * 100)) + str(maxlen) + '.csv'
    #df.to_csv(filename, index=False)
    #savefen(df, model_type, topp, maxlen,"rand")

    counts = count_moves(df)
    return counts

# Generate moves from random position from training data
# Generate random Position from Games
def generate_games_file(model_type, topp, maxlen, games_per_position, depth, file, position_per_file):

    tokenizer = GPT2TokenizerFast.from_pretrained("data/fentokenizer")
    model = AutoModelForCausalLM.from_pretrained("data/model_" + model_type)

    # Read FEN file
    with open(file, 'r') as f:
        lines = f.readlines()
    df = pd.DataFrame(lines)

    outputlist = []

    for i in range(position_per_file):
        # Generate random Position from Games
        gamestring = str(df.iloc[randrange(df.shape[0])][0])
        gamelist = gamestring.split(';')
        start = ';'.join(gamelist[:depth])
        input_ids = tokenizer.encode(start, return_tensors='pt')

        # generate text until the output length (which includes the context length) reaches max_length
        for el in range(games_per_position):
            output = model.generate(input_ids,
                                    max_length=maxlen,
                                    top_p=topp,
                                    do_sample=True)
            move = tokenizer.decode(output[0], skip_special_tokens=True)
            outputlist.append(move)

    df = pd.DataFrame(outputlist, columns=["Move"])
    #filename = 'data/games/games_file_' + model_type + str(int(topp * 100)) + str(maxlen) + '.csv'
    #df.to_csv(filename, index=False)
    #savefen(df, model_type, topp, maxlen,"file")

    counts = count_moves(df)
    return counts