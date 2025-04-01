import random
from queue import Queue

import matplotlib.pyplot as plt
import numpy as np


def create_maze(dim):
    """
    Genera un laberinto utilizando el algoritmo Hunt and Kill.
    
    :param dim: Dimensión del laberinto (dim x dim).
    :return: Laberinto generado como una matriz numpy.
    """
    if dim <= 0:
        raise ValueError("La dimensión debe ser un número positivo.")

    # Inicializar el laberinto con paredes
    maze = initialize_maze(dim)
    current_cell = (0, 0)
    maze[2 * current_cell[0] + 1, 2 * current_cell[1] + 1] = 0  # Marcar la celda inicial como visitada

    # Inicializar visualización
    fig, ax = initialize_visualization(maze)

    # Lista de celdas no visitadas
    unvisited_cells = [(x, y) for x in range(dim) for y in range(dim) if maze[2 * x + 1, 2 * y + 1] == 1]

    # Fases de caminata y caza
    while unvisited_cells:
        current_cell = walk(maze, current_cell, unvisited_cells, ax) or hunt(maze, unvisited_cells, ax)
    
    # Crear entrada y salida
    maze[1, 0] = 0
    maze[-2, -1] = 0

    # Actualizar visualización final
    update_visualization(ax.images[0], maze)
    plt.pause(0.1)

    return maze


def initialize_maze(dim):
    """
    Inicializa una cuadrícula llena de paredes.
    
    :param dim: Dimensión del laberinto.
    :return: Matriz numpy que representa el laberinto inicial.
    """
    return np.ones((dim * 2 + 1, dim * 2 + 1), dtype=int)


def initialize_visualization(maze):
    """
    Inicializa la visualización del laberinto.
    
    :param maze: Laberinto inicial.
    :return: Figura y eje de matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(maze, cmap='binary', interpolation='nearest')
    plt.pause(0.1)
    return fig, ax


def walk(maze, current_cell, unvisited_cells, ax):
    """
    Fase de caminata: realiza un paseo aleatorio desde la celda actual.
    
    :param maze: Laberinto generado.
    :param current_cell: Celda actual.
    :param unvisited_cells: Lista de celdas no visitadas.
    :param ax: Eje de matplotlib para la visualización.
    :return: Nueva celda actual o None si no se puede avanzar.
    """
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Direcciones posibles (arriba, abajo, izquierda, derecha)
    random.shuffle(directions)  # Mezclar las direcciones para aleatorizar el paseo

    for dx, dy in directions:  # Iterar sobre cada dirección
        nx, ny = current_cell[0] + dx, current_cell[1] + dy  # Calcular la nueva celda
        if 0 <= nx < maze.shape[0] // 2 and 0 <= ny < maze.shape[1] // 2:  # Verificar límites del laberinto
            if maze[2 * nx + 1, 2 * ny + 1] == 1:  # Verificar si la nueva celda no está visitada
                # Tallar la pared hacia la nueva celda
                maze[2 * current_cell[0] + 1 + dx, 2 * current_cell[1] + 1 + dy] = 0
                maze[2 * nx + 1, 2 * ny + 1] = 0  # Marcar la nueva celda como visitada
                unvisited_cells.remove((nx, ny))  # Eliminar la nueva celda de la lista de no visitadas
                update_visualization(ax.images[0], maze)  # Actualizar la visualización
                return (nx, ny)  # Devolver la nueva celda actual
    return None  # Si no se puede avanzar, devolver None


def hunt(maze, unvisited_cells, ax):
    """
    Fase de caza: busca una celda no visitada con vecinos visitados.
    
    :param maze: Laberinto generado.
    :param unvisited_cells: Lista de celdas no visitadas.
    :param ax: Eje de matplotlib para la visualización.
    :return: Nueva celda actual o None si no se encuentra ninguna.
    """
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Direcciones posibles (arriba, abajo, izquierda, derecha)

    for hunt_x, hunt_y in unvisited_cells[:]:  # Iterar sobre una copia de la lista de celdas no visitadas
        vecinos_visitados = [
            (adj_x, adj_y) for dx, dy in directions  # Para cada dirección
            for adj_x, adj_y in [(hunt_x + dx, hunt_y + dy)]  # Calcular coordenadas del vecino
            if 0 <= adj_x < maze.shape[0] // 2 and 0 <= adj_y < maze.shape[1] // 2  # Verificar límites del laberinto
            and maze[2 * adj_x + 1, 2 * adj_y + 1] == 0  # Verificar si el vecino está visitado
        ]
        if vecinos_visitados:  # Si hay vecinos visitados
            adj_x, adj_y = random.choice(vecinos_visitados)  # Elegir un vecino visitado aleatorio
            dir_x, dir_y = hunt_x - adj_x, hunt_y - adj_y  # Calcular la dirección hacia el vecino
            maze[2 * adj_x + 1 + dir_x, 2 * adj_y + 1 + dir_y] = 0  # Tallar la pared entre la celda no visitada y el vecino
            maze[2 * hunt_x + 1, 2 * hunt_y + 1] = 0  # Marcar la celda no visitada como visitada
            unvisited_cells.remove((hunt_x, hunt_y))  # Eliminar la celda de la lista de no visitadas
            update_visualization(ax.images[0], maze)  # Actualizar la visualización
            return (hunt_x, hunt_y)  # Devolver la nueva celda actual
    return None  # Si no se encuentra ninguna celda no visitada con vecinos visitados


def update_visualization(img, data):
    """
    Actualiza la visualización del laberinto.
    
    :param img: Imagen de matplotlib.
    :param data: Datos para actualizar la imagen.
    """
    img.set_data(data)
    plt.pause(0.01)


def find_path(maze):
    """
    Encuentra el camino más corto desde la entrada hasta la salida utilizando BFS.
    
    :param maze: Laberinto generado.
    :return: Lista de coordenadas que representan el camino.
    """
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = (1, 1)
    end = (maze.shape[0] - 2, maze.shape[1] - 2)
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True
    queue = Queue()
    queue.put((start, []))

    fig, ax = initialize_visualization(maze)

    while not queue.empty():
        (node, path) = queue.get()
        for dx, dy in directions:
            next_node = (node[0] + dx, node[1] + dy)
            if next_node == end:
                return path + [next_node]
            if (0 <= next_node[0] < maze.shape[0] and 0 <= next_node[1] < maze.shape[1]
                    and maze[next_node] == 0 and not visited[next_node]):
                visited[next_node] = True
                queue.put((next_node, path + [next_node]))
                update_visualization(ax.images[0], visited)
    return None


def draw_maze(maze, path=None):
    """
    Dibuja el laberinto y el camino (si se proporciona).
    
    :param maze: Laberinto generado.
    :param path: Camino desde la entrada hasta la salida.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    if path:
        x_coords = [x[1] for x in path]
        y_coords = [y[0] for y in path]
        ax.plot(x_coords, y_coords, color='red', linewidth=2)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.arrow(0, 1, 0.4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
    ax.arrow(maze.shape[1] - 1, maze.shape[0] - 2, 0.4, 0, fc='blue', ec='blue', head_width=0.3, head_length=0.3)
    plt.show()


if __name__ == "__main__":
    try:
        dim = int(input("Ingrese la dimensión del laberinto: "))
        maze = create_maze(dim)
        path = find_path(maze)
        draw_maze(maze, path)
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")