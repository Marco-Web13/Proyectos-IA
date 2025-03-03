# 8 puzzle solver
import heapq

puzzleInicial= [
    [5, 3, 2],
    [1, 4, 6],
    [8, 7, 0]
]

puzzleFinal=[  [1, 2, 3], 
               [4, 5, 6], 
               [7, 8, 0] ]

def printPuzzle(puzzle):
    for i in range(3):
        for j in range(3):
            print(puzzle[i][j], end=' ')
        print()
    print()

def moveUp(puzzle):
    i, j = findZero(puzzle)
    if i > 0:
        puzzle[i][j], puzzle[i-1][j] = puzzle[i-1][j], puzzle[i][j]
    
def moveDown(puzzle):
    i, j = findZero(puzzle)
    if i < 2:
        puzzle[i][j], puzzle[i+1][j] = puzzle[i+1][j], puzzle[i][j]

def moveLeft(puzzle):
    i, j = findZero(puzzle)
    if j > 0:
        puzzle[i][j], puzzle[i][j-1] = puzzle[i][j-1], puzzle[i][j]

def moveRight(puzzle):
    i, j = findZero(puzzle)
    if j < 2:
        puzzle[i][j], puzzle[i][j+1] = puzzle[i][j+1], puzzle[i][j]

def findZero(puzzle):
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] == 0:
                return i, j
            
def movePice(puzzle, movement):
    new_puzzle = [row.copy() for row in puzzle]
    if movement == "Up":
        moveUp(new_puzzle)
        return new_puzzle
    elif movement == "Down":
        moveDown(new_puzzle)
        return new_puzzle
    elif movement == "Left":
        moveLeft(new_puzzle)
        return new_puzzle
    elif movement == "Right":
        moveRight(new_puzzle)
        return new_puzzle
    return None

class Node:
    def __init__(self, puzzle, movimiento, costo , heuristica, parent):
        self.puzzle = puzzle
        self.movimiento = movimiento
        self.costo = costo
        self.heuristica = heuristica
        self.parent = parent
        
    def __lt__(self, other):
        return (self.costo + self.heuristica) < (other.costo + other.heuristica)

def CalcularHeuristica(puzzle):
    heuristica = 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != puzzleFinal[i][j]:
                i2, j2 = getPosicion(puzzle[i][j])
                heuristica += (abs(i - i2) + abs(j - j2))
    return heuristica

def getPosicion(valor):
    for i in range(3):
        for j in range(3):
            if puzzleFinal[i][j] == valor:
                return i,j

def algoritmo_a_star(puzzle):
    nodosVisitados = set()
    cola = []
    heapq.heappush(cola, Node(puzzle, "", 0, CalcularHeuristica(puzzle), None))
    a = 0
    while cola:
        a += 1
       
        actual = heapq.heappop(cola)
        if actual.puzzle == puzzleFinal:
            break
        nodosVisitados.add(str(actual.puzzle))
        movimientos = ["Up", "Down", "Left", "Right"]
        for movimiento in movimientos:
            siguiente = movePice(actual.puzzle, movimiento)
            if siguiente is not None and str(siguiente) not in nodosVisitados:

                heapq.heappush(cola, Node(siguiente, movimiento, actual.costo + 1, CalcularHeuristica(siguiente), actual))
    
    recorrido = []
    while actual is not None:
        recorrido.append(actual)
        actual = actual.parent
    recorrido.reverse()

    print("Movimientos:")
    for nodo in recorrido:
        if nodo.movimiento:
            print('Movimiento: ' + nodo.movimiento)
        for row in nodo.puzzle:
            for num in row:
                print('| '+ str(num), end=' |')
            print()


    print("Cantidad de Movimientos: ",len(recorrido)-1)
    print('')

print("Puzzle Inicial")
printPuzzle(puzzleInicial)
print("Solucion")
algoritmo_a_star(puzzleInicial)

