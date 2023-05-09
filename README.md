# Chess-GPT2
## Code for the Bachelor's thesis "Evaluating Transformer Architecture for The Game of Chess" by Raiko Marrandi.

All of the runnable methods are implemented in main.py.

The general structure of the program-tree:

- Main
  - Train Tokenizer
  - Train Model
  - For each saved/testing state of the model:
    - Test accuracy on games generated from:
      1. Typical opening positions
      2. Positions resulting from *n* random moves
      3. The *n*-th position from a game dataset
    - Calculate perplexity
    - Save accuracy and perplexity scores

Data for the project can be sourced from database.lichess.org and extracted with pgn-extract by David J. Barnes (https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/). 
PGN data should be extracted to one line per game and all annotations should be removed, this can be done with the command pgn-extract --notags --nomovenumbers --noresults -w1000 -C -N -V -bl20 input.pgn --output output.pgn

FEN data should instead be converted with pgnToFen https://github.com/RMarrandi/pgnToFen.

The code will work almost out-of-the-box, needing only filepaths to be adjusted.
