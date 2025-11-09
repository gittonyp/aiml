def print_board(b):
    for r in b:
        print(" | ".join(r))
        print("-----")

def check_winner(b):
    for r in b:
        if r[0] == r[1] == r[2] != " ":
            return r[0]
    for c in range(3):
        if b[0][c] == b[1][c] == b[2][c] != " ":
            return b[0][c]
    if b[0][0] == b[1][1] == b[2][2] != " ":
        return b[0][0]
    if b[0][2] == b[1][1] == b[2][0] != " ":
        return b[0][2]
    return None

board = [[" "]*3 for _ in range(3)]
player = "X"
turns = 0

while turns < 9:
    print_board(board)
    try:
        r = int(input(f"Enter row (0-2) for {player}: "))
        c = int(input(f"Enter col (0-2) for {player}: "))
        
        if not (0 <= r <= 2 and 0 <= c <= 2):
            print("Coordinates out of bounds. Try again.")
            continue
            
        if board[r][c] == " ":
            board[r][c] = player
            turns += 1
            winner = check_winner(board)
            if winner:
                print_board(board)
                print(f"Winner is {winner}")
                break
            player = "O" if player == "X" else "X"
        else:
            print("Cell taken, try again")
    except ValueError:
        print("Invalid input. Please enter a number (0-2).")

else:
    print_board(board)
    print("It's a Draw!")