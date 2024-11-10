ans_arr = [[0,5,0,6,0,3,6,0,0,0,7,4],
           [0,7,7,1,0,0,5,0,4,6,5,0],
           [0,0,0,7,5,3,5,0,6,0,6,0],
           [6,4,3,0,0,7,0,5,7,1,0,0],
           [7,0,0,0,2,7,4,2,0,0,0,7],
           [2,0,6,6,6,0,0,6,3,7,0,4],
           [5,4,4,0,7,0,0,7,0,6,2,5],
           [7,0,0,0,1,5,6,1,0,0,7,0],
           [0,5,3,7,0,5,0,0,5,0,4,6],
           [6,5,0,0,0,7,2,0,6,0,0,5],
           [0,6,7,3,0,0,4,6,6,4,0,0],
           [0,0,0,4,6,3,7,0,0,3,7,0]]

def is_valid_cell(x, y, nrows, ncols, arr):
    return x >= 0 and x < nrows and y >= 0 and y < ncols and arr[x][y] == 0

def dfs(x, y, visited, nrows, ncols, arr):
    visited[x][y] = True
    area = 1
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if is_valid_cell(nx, ny, nrows, ncols, arr) and not visited[nx][ny]:
            area += dfs(nx, ny, visited, nrows, ncols, arr)
    return area

def calculate_product_of_areas(arr):
    nrows, ncols = len(arr), len(arr[0])
    product = 1
    area_arr = []
    visited = [[False] * ncols for _ in range(nrows)]
    for i in range(nrows):
        for j in range(ncols):
            if arr[i][j] == 0 and not visited[i][j]:
                area = dfs(i, j, visited, nrows, ncols, arr)
                if area != 1: area_arr.append(area)
                product *= area
    print(area_arr)
    return product

print(calculate_product_of_areas(ans_arr))