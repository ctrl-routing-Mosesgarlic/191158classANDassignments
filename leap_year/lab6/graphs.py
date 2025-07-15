class Graphs:
    def __init__(self, directed=False):
        self.directed = directed
        self.adj_list = dict()    
    
    def __repr__(self):
        str_graph = ""
        
        for key, value in self.adj_list.items():
            str_graph += f"{key} -> {value}\n"
            
        return str_graph
    

        
        
    def dfs(self, start):    
        """We use STACK(lifo) FOR DFS, as helping data structures"""    
        visited = set()  # set of visited nodes    
        stack = [start]  # nodes to be processed, so have a starting node    
        order = []  # Our result in order, which is traversal order    
        while stack:        
            # while we still have elements to process,        
            """get the elements to process in lifo way"""        
            node = stack.pop()  # last element added        
            if node not in visited:  # if node is not visited            
                visited.add(node)  # first add it to the visited nodes            
                order.append(node)  # append it to our resulting list            
                # Then get all the neighbours of this node, use our available function            
                neighbours = self.get_neighbours(node)            
                # then iterate through them            
                for neighbour in sorted(neighbours, reverse=True):                
                    # means we have a weighted connection not just individual value                
                    if isinstance(neighbour, tuple):                    
                        neighbour = neighbour[0]  # just take the value and leave the weight                    
                    # if neighbour is not visited, append it to stack                
                    if neighbour not in visited:                    
                        stack.append(neighbour)    
        return order
    
    def bfs(self, start):    
        """We use QUEUES(fifo) IN BFS, AND STACK(lifo) FOR DFS, as helping data structures"""    
        visited = set() #set of visited nodes    
        queue = [start] #nodes to be processed, so have a starting node    
        order = [] #Our result in order, which is traversal order    
        while queue:        
            # while we still have elements to process,        
            """get the elements to process in fifo way"""        
            node = queue.pop(0) #first element added        
            if node not in visited: #if node is not visited            
                visited.add(node) #first add it to the visited nodes            
                order.append(node) #append it to our resulting list            
                #Then get all the neighbours of this node, use our available function            
                neighbours = self.get_neighbours(node)            
                # then iterate through them            
                for neighbour in neighbours:                
                    #means we have a weighted connection not just individual value                
                    if isinstance(neighbour, tuple):                    
                        neighbour = neighbour[0]  #just take the value and leave the weight                    
                    # if neighbour is not visited, append it to queue                
                    if neighbour not in visited:                    
                        queue.append(neighbour)    
        return order


    
    def add_node(self):
        if node not in self.adj_list:
            self.adj_list[node] = set()
        else:
            raise ValueError("Node already exists in the graph.")
        
    def add_edge(self, source_node, destination_node, weighted=None):
        if source_node not in self.adj_list:
            self.add_node(source_node)
            
        if destination_node not in self.adj_list:
            self.add_node(destination_node)
            
        if weighted is not None:
            self.adj_list[source_node].add((destination_node, weighted))
            if self.directed:
                self.adj_list[destination_node].add((source_node, weighted))
        else:
            self.adj_list[source_node].add(destination_node)
            if self.directed:
                self.adj_list[destination_node].add(source_node)
        
        def obtain_neighbors(self, node):
        
        def adj_matrix(self):
            pass
    
    if __name__ == "__main__":
        graph.obj = Graphs()
        print("Graph object created:", graph.obj)
        
# class Graphs:
#     def _init_(self, directed):
#         self.directed = directed
#         self.adj_list = dict()

#     def _repr_(self):
#         str_graph = ""
#         for key, value in self.adj_list.items():
#             str_graph += f"{key} -> {value}\n"
#         return str_graph.strip()

#     def bfs(self, start):    
#         """We use QUEUES(fifo) IN BFS, AND STACK(lifo) FOR DFS, as helping data structures"""    
#         visited = set() #set of visited nodes    
#         queue = [start] #nodes to be processed, so have a starting node    
#         order = [] #Our result in order, which is traversal order    
#         while queue:        
#             # while we still have elements to process,        
#             """get the elements to process in fifo way"""        
#             node = queue.pop(0) #first element added        
#             if node not in visited: #if node is not visited            
#                 visited.add(node) #first add it to the visited nodes            
#                 order.append(node) #append it to our resulting list            
#                 #Then get all the neighbours of this node, use our available function            
#                 neighbours = self.get_neighbours(node)            
#                 # then iterate through them            
#                 for neighbour in neighbours:                
#                     #means we have a weighted connection not just individual value                
#                     if isinstance(neighbour, tuple):                    
#                         neighbour = neighbour[0]  #just take the value and leave the weight                    
#                     # if neighbour is not visited, append it to queue                
#                     if neighbour not in visited:                    
#                         queue.append(neighbour)    
#         return order

#     def dfs(self, start):    
#         """We use STACK(lifo) FOR DFS, as helping data structures"""    
#         visited = set()  # set of visited nodes    
#         stack = [start]  # nodes to be processed, so have a starting node    
#         order = []  # Our result in order, which is traversal order    
#         while stack:        
#             # while we still have elements to process,        
#             """get the elements to process in lifo way"""        
#             node = stack.pop()  # last element added        
#             if node not in visited:  # if node is not visited            
#                 visited.add(node)  # first add it to the visited nodes            
#                 order.append(node)  # append it to our resulting list            
#                 # Then get all the neighbours of this node, use our available function            
#                 neighbours = self.get_neighbours(node)            
#                 # then iterate through them            
#                 for neighbour in sorted(neighbours, reverse=True):                
#                     # means we have a weighted connection not just individual value                
#                     if isinstance(neighbour, tuple):                    
#                         neighbour = neighbour[0]  # just take the value and leave the weight                    
#                     # if neighbour is not visited, append it to stack                
#                     if neighbour not in visited:                    
#                         stack.append(neighbour)    
#         return order

#     def add_node(self, node):
#         if node not in self.adj_list:
#             self.adj_list[node] = set()
#         else:
#             raise ValueError("Node exist")

#     def add_edge(self, source_node, destination_node, weighted=None):
#         if source_node not in self.adj_list:
#             self.add_node(source_node)
        
#         if destination_node not in self.adj_list:
#             self.add_node(destination_node)

#         if weighted is None:
#             self.adj_list[source_node].add(destination_node)
#             if not self.directed:
#                 self.adj_list[destination_node].add(source_node)
#         else:
#             self.adj_list[source_node].add((destination_node, weighted))
#             if not self.directed:
#                 self.adj_list[destination_node].add((source_node, weighted))

#     def get_neighbours(self, key_node):
#         return self.adj_list.get(key_node, set())

#     def adj_matrix(self):
#         pass

# if _name_ == '_main_':
#     graphy_obj = Graph(directed=False)
#     graphy_obj.add_node('A')
#     graphy_obj.add_node('B')
#     graphy_obj.add_node('C')
#     graphy_obj.add_edge('A', 'B', weighted=1)
#     graphy_obj.add_edge('B', 'C')
#     print(graphy_obj)
#     print("BFS traversal starting from A:", graphy_obj.bfs('A'))
#     print("DFS traversal starting from A:", graphy_obj.dfs('A'))