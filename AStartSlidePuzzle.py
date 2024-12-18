from selenium.webdriver.common.by import By
import time
from datetime import datetime
import math
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.service import Service



def main():
    service = Service()

    options = webdriver.ChromeOptions()

    options.add_extension('/Users/adamsalonen/Documents/GitHub/PythonApps/SlidePuzzle/adblockult.crx')

    driver = webdriver.Chrome(service=service, options=options)

    driver.get('https://sliding.toys/mystic-square/8-puzzle/')   

    while True:
        if driver.window_handles.__len__() > 1:
            break
        else:
            print('loading...')

    windows = driver.window_handles

    driver.switch_to.window(windows[1])

    driver.close()

    driver.switch_to.window(windows[0])

    table = driver.find_element(By.ID, 'table')

    tiles = table.find_elements(By.CLASS_NAME, 'cell')

    sqr_root = math.floor(math.sqrt(tiles.__len__() + 1))

    bound = sqr_root - 1

    listCounter = 0

    global main_best_h
    main_best_h = 1000000

    complete = True
    for y in range(bound):
        for x in range(bound):
            myString = y.__str__() + '-' + x.__str__()
            myTile = tiles[listCounter].get_attribute('data-position')
            listCounter += 1

            if myTile != myString:
                complete = False

    matrix = np.zeros((sqr_root,sqr_root))
    if complete != True:
        for x in tiles:
            value = int(x.get_attribute('data-value'))
            position = x.get_attribute('data-position')

            currentX = int(position[2])
            currentY = int(position[0])

            #fill the matrix
            matrix[currentY, currentX] = x.get_attribute('data-value')

            current_cost = find_cost(matrix, sqr_root)

    root = Node(matrix=matrix, tile_num=None, cost=current_cost)

    root.set_explored()

    global visited_nodes
    visited_nodes = {root.matrix.__str__() : root}

    setback_counter = 0

    leaf_list = [root]

    start = datetime.now().timestamp()
    while True:       
        current_root = leaf_list[0]

        current_root.set_explored()

        leaf_list.remove(current_root)

        current_matrix = current_root.matrix

        if current_root.individual_cost == 0:
            break
        
        zero_index = np.argwhere(current_matrix == 0)[0]

        clickable = adjacent_indicies(zero_index, sqr_root)

        children_list = []
        for i in range(0, len(clickable)):
            child_matrix = np.copy(current_matrix)
            value = int(child_matrix[clickable[i][0], clickable[i][1]])
            child_matrix[zero_index[0], zero_index[1]] = value
            child_matrix[clickable[i][0], clickable[i][1]] = 0

            children_list.append(Node(matrix=child_matrix, tile_num=value, cost=None))         

        children_costs = []
        for x in children_list:
            children_costs.append(find_cost(x.matrix, sqr_root))

        setback = True
        for x in range(len(children_list)):
            children_list[x].cost = children_costs[x]
            current_node = children_list[x]
            try:
                visited_nodes[current_node.matrix.__str__()]
                continue
                
            except:
                current_root.add_child(child_matrix=current_node.matrix, child_tile_num=current_node.tile_num, 
                                       child_cost=current_node.cost)
                setback = False

        if not setback:
            for child in current_root.children:
                insert_into_sorted_array(leaf_list, child)
            setback_counter += 1
                
        for child in current_root.children:
            if child.individual_cost < main_best_h:
                main_best_h =  child.individual_cost
                print('current best h: {}'.format(main_best_h))

        



    end = datetime.now().timestamp()

    print('total time to find best path: {}ms'.format(end - start))

    leaves = []

    inorder_traversal(root, leaves)

    final_path = sorted(leaves, key=lambda x: x.individual_cost)[0].path_to_root()

    path_values = []
    for x in final_path:
        path_values.append(x.tile_num)

    for x in path_values:
        if x is not None:
            time.sleep(0.005)
            driver.find_element(By.CSS_SELECTOR, 'button[data-value="{}"]'.format(x)).click()

    print('nodes searched: {}'.format(len(visited_nodes)))
    print('final depth: {}'.format(len(path_values)))
    print('Press enter to continue...')

    input()

    driver.quit()


def insert_into_sorted_array(arr, value):
    low = 0
    high = len(arr) - 1

    # Perform binary search to find the insertion index
    while low <= high:
        mid = (low + high) // 2
        if arr[mid].cost < value.cost:
            low = mid + 1
        else:
            high = mid - 1

    # Insert the value at the correct position
    arr.insert(low, value)

def inorder_traversal(node, leaves):
    if not node.children:  # Check if the node has no children (i.e., it's a leaf)
        leaves.append(node)
    else:
        for child in node.children:
            inorder_traversal(child, leaves)



def find_cost(matrix, root):
    cost = 0
    for i in range(root):
        for j in range(root):
            if matrix[j, i] != 0:
                desiredX = xval(matrix[j,i], root)
                desiredY = yval(matrix[j,i], root)

                currentX = i
                currentY = j

                cost += abs(desiredX - currentX) + abs(desiredY - currentY)

    return cost


def adjacent_indicies(zero_index, root):    
    adjacent_indices = []

    adjacent_indices.append([zero_index[0] + 1, zero_index[1]])
    adjacent_indices.append([zero_index[0] - 1, zero_index[1]])
    adjacent_indices.append([zero_index[0], zero_index[1] + 1])
    adjacent_indices.append([zero_index[0], zero_index[1] - 1])

    adjacent_indices_sublist = []

    for x in adjacent_indices:
        if root not in x and -1 not in x:  
            adjacent_indices_sublist.append(x)

    return adjacent_indices_sublist

def xval(n, root):
    if n % root == 0:
        return root - 1

    return int(n % root - 1)

def yval(n, root):
    if n % root == 0:
        return int(n / root - 1)

    return int(n / root)


class Node:
    def __init__(self, matrix, tile_num, cost):
        self.matrix = matrix
        self.tile_num = tile_num
        self.cost = cost
        self.children = []
        self.parent = None
        self.explored = False
        self.depth = 0
        self.individual_cost = None

    def __hash__(self):
        key = self.matrix.__str__() + self.cost.__str__()
        return hash(key)
        
    def add_child(self, child_matrix, child_tile_num, child_cost):
        new_child = Node(matrix=child_matrix, tile_num=child_tile_num, cost=(child_cost + self.cost))
        new_child.individual_cost = child_cost
        new_child.parent = self
        self.children.append(new_child)
        new_child.depth = self.depth + 1
        visited_nodes.update({new_child.matrix.__str__() : new_child})
        return new_child

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self
        if node is not None:
            print("  " * level + str(node.value[1]))
            for child in node.children:
                self.print_tree(child, level + 1)


    def contains_value(self, target):
        if (self.value[0][0] == target[0][0]).all():
            return True
        for child in self.children:
            if child.contains_value(target):
                return True
        return False
    
    def set_explored(self):
        self.explored = True
    
    def path_to_root(self):
        path = [self]  # Start with the current node
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.parent)
            current_node = current_node.parent
        return path[::-1]
    
    def find_root(self):
        current_node = self
        while current_node.parent is not None:
            current_node = current_node.parent
        return current_node

if __name__ == "__main__":
    main()