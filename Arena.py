


import logging

from tqdm import tqdm
import numpy as np
import torch
import math
from Game import Game #used to be relative import: from .Game import Game

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.last_row_captured_count = {'1' : [], '-1': []}

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board,curPlayer)

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
            # print('valids=', valids)
            # print('other(actual) valids=', self.game.getValidMoves(board,curPlayer))
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer),valids)


            print('action chosen=',action)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, curPlayer, self.last_row_captured_count = self.game.getNextState(board, curPlayer, action, self.last_row_captured_count)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board,curPlayer)
        print('Game end at perpective of player',curPlayer)
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws


def random_agent(board,valid_moves):
    options = []
    for i, action_valid in enumerate(valid_moves):
        if action_valid == 1:
            options.append(i)

    action = np.random.choice(options)
    return action

valids = torch.tensor([0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
        0., 0., 0., 1., 1., 0.])

# for _ in range(50):
#     print(random_agent(None,valids))

game = Game()

# for i in range(24):
#     print(f'neighbor for {i}',game._getNeighbor(game.getInitBoard()[0],i))
arena = Arena(player1=random_agent,player2=random_agent,game=Game(),display=game.printBoard)
#
# arena.playGame(verbose=True)
oneWon, twoWon, draw = arena.playGames(num=1,verbose=True)
print('-------')
print('stats:')
print('oneWon:',oneWon)
print('twoWon:',twoWon)
print('draws:',draw)
# board = (torch.tensor([1,1,1,1,1]),torch.tensor([0,0,0]))
# board_ = [torch.clone(board[0]),torch.clone(board[1])]
#
# board_[0] = -1 * board_[0]
#
# print(board_)
# print(board)
