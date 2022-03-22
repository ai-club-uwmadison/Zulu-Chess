import torch
import math

class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self):
        pass

    def _token(self,i):
        if i == 1:
            return 'O'
        elif i == -1:
            return 'X'
        else:
            return '0'

    def printBoard(self,board,player):
        board_only = board[0]
        print(self._token(int(board_only[0,0].item())),end='-'*30)
        print(self._token(int(board_only[0,1].item())),end='-'*30)
        print(self._token(int(board_only[0,2].item())))
        print('-'*10,end='')
        print(self._token(int(board_only[1,0].item())),end='-'*20)
        print(self._token(int(board_only[1,1].item())),end='-'*20)
        print(self._token(int(board_only[1,2].item())),end='-'*10 + '\n')
        print('-'*20,end='')
        print(self._token(int(board_only[2,0].item())),end='-'*10)
        print(self._token(int(board_only[2,1].item())),end='-'*10)
        print(self._token(int(board_only[2,2].item())),end='-'*20+'\n')

        print(self._token(int(board_only[0,7].item())),end='-'*9)
        print(self._token(int(board_only[1,7].item())),end='-'*9)
        print(self._token(int(board_only[2,7].item())),end='-'*21)
        print(self._token(int(board_only[2,3].item())),end='-'*9)
        print(self._token(int(board_only[1,3].item())),end='-'*9)
        print(self._token(int(board_only[0,3].item())))
        print('-'*20,end='')
        print(self._token(int(board_only[2,6].item())),end='-'*10)
        print(self._token(int(board_only[2,5].item())),end='-'*10)
        print(self._token(int(board_only[2,4].item())),end='-'*20 + '\n')
        print('-'*10,end='')
        print(self._token(int(board_only[1,6].item())),end='-'*20)
        print(self._token(int(board_only[1,5].item())),end='-'*20)
        print(self._token(int(board_only[1,4].item())),end='-'*10 + '\n')
        print(self._token(int(board_only[0,6].item())),end='-'*30)
        print(self._token(int(board_only[0,5].item())),end='-'*30)
        print(self._token(int(board_only[0,4].item())))
        print()
        data = board[1]
        print('player 1 (O) remaining cows: ',int(data[0].item()))
        print('player -1 (X) remaining cows:',int(data[1].item()))
        print('player 1 (O) just captured row?',int(data[2].item()))
        print('player -1 (X) just captured row?',int(data[3].item()))
        print('player 1 (O) cow selection:',int(data[4].item()))
        print('player -1 (X) cow selection:',int(data[5].item()))
        print('player 1 (O) placeable cows: ',int(data[6].item()))
        print('player -1 (X) placeable cows:',int(data[7].item()))
        print('game state: ', 'place tokens' if data[8] == 0 else 'move/fly tokens')
        print('available actions:', self.getValidMoves(board,player))
        print('game end?', self.getGameEnded(board,player))
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        board = torch.zeros((3,8))
        # board[0,0] = 1
        # board[0,1] = -1
        # board[0,2] = -1
        # board[0,3] = -1
        # board[0,4] = -1
        # board[0,6] = 1
        # board[0,7] = 1
        # board[1,0] = 1
        # board[1,1] = 1
        # board[1,2] = -1
        # board[1,3] = 1
        # board[1,4] = -1
        # board[1,5] = 1
        # board[1,6] = -1
        # board[1,7] = 1
        # board[2,0] = 1
        # board[2,6] = 1
        # board[2,7] = -1

        #3 squares of 3x3 (innermost, middle, outermost)
        metadata = torch.tensor([12,12,0,0,-1,-1,12,12,0])
        #vector that represents each player's remaining tokens (alive, not necessarily placeable) + whether each player has just captured a row + the cow selected for moving for each player + the remaining cow tokens available to place for each players + the game phase code
        return board, metadata

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.getInitBoard()[0].size()

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return torch.prod(torch.tensor(self.getInitBoard()[0].size()))

    def getNextState(self, board, player, action, last_row_captured_count):
        """
        Input:
            board: current board with metadata
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        board_only = board[0]
        data = board[1]
        row_just_captured = data[ 2 if player == 1 else 3] == 1
        can_fly = data[0 if player == 1 else 1] <= 3 or data[-1] == 0
        cow_selected = data[4 if player == 1 else 5]
        player_index = 6 if player == 1 else 7
        game_state = data[-1]


        if action <= 23:
            if not row_just_captured and cow_selected == -1 and game_state == 0:
                #placing cow on the board
                square = min(math.ceil((action+1) / 8),3) #1st, 2nd, or 3rd square
                cell = action - 8 * (square-1)
                board_only[square-1,cell] = player
                #decrement placeable count
                data[player_index] -= 1
                print('PLACED cow on the board')

            if not row_just_captured and game_state == 1 and cow_selected == -1:
                #selecting cow to move
                data[4 if player == 1 else 5] = action
                print('SELECTED cow on the board')

                return board, player, last_row_captured_count
            if row_just_captured:
                # displace opponent's cow on the board
                action = action - 24
                square = min(math.ceil((action+1) / 8),3)
                cell = action - 8 * (square - 1)
                board_only[square-1, cell] = 0
                # decrement the other player's remaining cows
                data[1 if player == 1 else 0] -= 1
                #set row_captured to 0
                data[2 if player == 1 else 3] = 0
                print('DISPLACED cow on the board')
                #update opponent's row counts
                row_capture_count = {str(-player): 0, str(player): last_row_captured_count[str(player)]}

                row_capture_count[str(-player)] = 0
                horizontal_vertical_patterns = [
                    [0, 1, 2],
                    [2, 3, 4],
                    [4, 5, 6],
                    [6, 7, 0]
                ]
                # check capture within square
                for i in range(3):
                    square = board_only[i]
                    for pattern in horizontal_vertical_patterns:
                        if square[pattern[0]] == -player and square[pattern[1]] == -player and square[
                            pattern[2]] == -player:
                            # row captured!

                            captured = True
                            row_capture_count[str(-player)] += 1
                # check captures across squares
                for i in range(8):
                    if board_only[0][i] == -player and board_only[1][i] == -player and board_only[2][i] == -player:
                        # row captured!

                        captured = True
                        row_capture_count[str(-player)] += 1

                return board, -player, row_capture_count
            if not row_just_captured and cow_selected != -1 and game_state == 1:
                #moving cow on the board
                square = min(math.ceil((cow_selected+1) / 8),3)
                cell = cow_selected - 8 * (square - 1)
                board_only[square-1, cell] = 0
                square = min(math.ceil((action+1) / 8),3)
                cell = action - 8 * (square - 1)
                board_only[square-1, cell] = player
                data[4 if player == 1 else 5] = -1
                print('MOVED cow on the board')

        #update game state
        if data[6] <= 0 and data[7] <= 0:
            #if both players do not have any more placeable tokens, then the game phase is updated to 1 (where only moving tokens are possible)
            data[-1] = 1
            print('Game state updated')

        #check for any row capture on current player's side
        horizontal_vertical_patterns = [
            [0,1,2],
            [2,3,4],
            [4,5,6],
            [6,7,0]
        ]
        captured = False
        row_capture_count = { str(player) : 0, str(-player) : last_row_captured_count[str(-player)]}

        row_capture_count[str(player)] = 0
        #check capture within square
        for i in range(3):
            square = board_only[i]
            for pattern in horizontal_vertical_patterns:
                if square[pattern[0]] == player and square[pattern[1]] == player and square[pattern[2]] == player:
                    #row captured!

                    captured = True
                    row_capture_count[str(player)] += 1
        #check captures across squares
        for i in range(8):
            if board_only[0][i] == player and board_only[1][i] == player and board_only[2][i] == player:
                #row captured!

                captured = True
                row_capture_count[str(player)] += 1


        print('total row capture count (for current player):',row_capture_count[str(player)], 'last row capture count (for current player):',last_row_captured_count[str(player)])
        if row_capture_count[str(player)] >= last_row_captured_count[str(player)] and captured:
            data[2 if player == 1 else 3] = 1
            print('ROW CAPTURED')
            return board, player, row_capture_count

        return board, -player, row_capture_count

    def _getNeighbor(self,board,cell):
        #given the a cell, return its neighboring (adjacent) cells id
        #create a dummy board...
        board_only = torch.zeros((3,8))
        #index it
        board_only[0] = torch.arange(0,8)
        board_only[1] = torch.arange(8,16)
        board_only[2] = torch.arange(16,24)
        square = min(math.ceil((cell+1)/8) , 3)
        cell = cell - 8 * (square-1)
        neighbors = []
        square = square-1
        #check for horizontal/vertical neighbors within its square
        neighbors.append(board_only[square,(cell+1) % 8])
        neighbors.append(board_only[square,(cell-1) % 8])
        #check for diagonal neighbors outside of its square
        square_neighbors = {0 : [1], 1: [0,2], 2 : [1]}
        try:
            for square_ in square_neighbors[square]:
                neighbors.append(board_only[square_,cell])

        except IndexError:
            pass

        neighbors = [int(e.item()) for e in neighbors]
        return torch.LongTensor(neighbors)



    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        board_only = board[0]
        data = board[1]
        action_vector = torch.zeros((self.getActionSize(),))
        #check if the player has just captured a row
        player_index = 2 if player == 1 else 3
        captured_row = data[player_index] == 1
        if captured_row:
            #if a row has just been captured...
            for i in range(24):
                square = min(math.ceil((i+1)/8),3)
                cell = (i) - 8 * (square-1)
                status = board_only[square-1,cell]
                #any cow on the cell where it is occupied by the OTHER player can be displaced
                if status == -player:
                    action_vector[i] = 1
        else:
            #if a row has not been captured, check whether the current player can fly
            player_index = 0 if player == 1 else 1
            can_fly = data[player_index] <= 3 or data[-1] == 0

            #game phase 0 (placing cow anywhere)
            if can_fly and data[-1] == 0:
                #if can fly, then any empty cell is available for the current player to place their cow on
                for i in range(24):
                    square = min(math.ceil((i+1)/8),3)
                    cell = i - 8 * (square - 1)

                    status = board_only[square-1, cell]
                    # print(square - 1, cell, i, status)
                    if status == 0:
                        action_vector[i] = 1
            #game phase 1 (moving cow anywhere)
            elif can_fly and data[-1] == 1:
                # first check if the player has selected a cow to move
                player_index = 4 if player == 1 else 5
                cow_selected = data[player_index] > 0
                if not cow_selected:
                    for i in range(24):
                        square = min(math.ceil((i+1)/8),3)
                        cell = i - 8 * (square - 1)
                        status = board_only[square-1, cell]
                        #if cow is not selected, the player must select a cow to move, and this can be any cell occupied by the current player
                        if status == player:
                            action_vector[i] = 1
                else:
                    #if cow has been selected, all empty cells are available
                    for i in range(24):
                        square = min(math.ceil((i+1)/8),3)
                        cell = i - 8 * (square - 1)
                        status = board_only[square-1, cell]
                        if status == 0:
                            action_vector[i] = 1

            else:
                #if cannot fly, then only the cells within one adjacent reach of the current player's cow are available

                #first check if the player has selected a cow to move
                player_index = 4 if player == 1 else 5
                cow_selected = data[player_index] > 0
                if not cow_selected:
                    for i in range(24):
                        square = min(math.ceil((i+1)/8),3)
                        cell = i - 8 * (square - 1)
                        status = board_only[square-1, cell]
                        #if cow is not selected, the player must select a cow to move, and this can be any cell occupied by the current player that has a valid neighbor to move to
                        if status == player:
                            neighbors = self._getNeighbor(board_only,i)
                            for neighbor in neighbors:
                                square = min(math.ceil((neighbor + 1) / 8), 3)
                                cell = neighbor - 8 * (square - 1)
                                if board_only[square - 1, cell] == 0:
                                    action_vector[i] = 1
                                    break


                else:
                    #if cow has been selected, its neighboring cells are available
                    cow_selected = data[player_index]
                    neighbors = self._getNeighbor(board_only,cow_selected)
                    for neighbor in neighbors:
                        # all empty neighbors == available action
                        square = min(math.ceil((neighbor + 1) / 8), 3)
                        cell = neighbor - 8 * (square - 1)
                        if board_only[square-1,cell] == 0:
                            action_vector[neighbor] = 1
        return action_vector









    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        data = board[1]
        remaining_tokens = data[0 if player == 1 else 1]
        remaining_tokens_other = data[1 if player == 1 else 0]

        win = remaining_tokens_other <= 2
        if win:
            return 1
        loss = remaining_tokens <= 2
        if loss:
            return -1

        return 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        board = [torch.clone(board[0]),torch.clone(board[1])]
        if player == 1:
            data = board[1]
            canonical_data = torch.tensor([data[0],data[2],data[4],data[6],data[8]])
            board[1] = canonical_data
            return board
        else:
            board[0] = board[0] * -1
            data = board[1]
            canonical_data = torch.tensor([data[1],data[3],data[5],data[7],data[8]])
            board[1] = canonical_data
            return board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        boardString = ""
        board_only = board[0]
        for i in range(24):
            square = min(math.ceil((i+1)/8),3)
            cell = i - 8 * (square - 1)
            boardString += str(board_only[square,cell])
        data = board[1]
        for datum in data:
            boardString += str(datum)

        return boardString



def play():
    game = Game()
    player = 1
    board = game.getInitBoard()
    board_only = board[0]
    last_row_captured_count = {'1' : 0, '-1': 0}
    # check for any row capture on current player's side
    horizontal_vertical_patterns = [
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 0]
    ]
    captured = False
    row_capture_count = {str(player): 0, str(-player): 0}

    row_capture_count[str(player)] = 0
    # check capture within square
    for i in range(3):
        square = board_only[i]
        for pattern in horizontal_vertical_patterns:
            if square[pattern[0]] == player and square[pattern[1]] == player and square[pattern[2]] == player:
                # row captured!

                captured = True
                row_capture_count[str(player)] += 1
    # check captures across squares
    for i in range(8):
        if board_only[0][i] == player and board_only[1][i] == player and board_only[2][i] == player:
            # row captured!

            captured = True
            row_capture_count[str(player)] += 1

    row_capture_count[str(-player)] = 0
    # check capture within square
    for i in range(3):
        square = board_only[i]
        for pattern in horizontal_vertical_patterns:
            if square[pattern[0]] == -player and square[pattern[1]] == -player and square[pattern[2]] == -player:
                # row captured!

                captured = True
                row_capture_count[str(-player)] += 1
    # check captures across squares
    for i in range(8):
        if board_only[0][i] == -player and board_only[1][i] == -player and board_only[2][i] == -player:
            # row captured!

            captured = True
            row_capture_count[str(-player)] += 1

    last_row_captured_count = row_capture_count

    while True:
        game.printBoard(board,player)
        print('player',player,'\'s turn')
        valid_moves = game.getValidMoves(board,player)
        if valid_moves.sum() < 0.001:
            print('no valid moves for player',player,'; skipping turn.')
            player = -player
            continue
        action = int(input('Type action: (0-24), from outer square to inner square clockwise from top left corner:\n'))
        if action > 24 or action < 0 or valid_moves[action] == 0:
            print('Invalid action! try again!')
            continue
        board, player, last_row_captured_count = game.getNextState(board,player,action,last_row_captured_count)
        end = game.getGameEnded(board,player)
        if end != 0:
            print('Game ended with a reward of ',end,'for player',player)
            break



play()