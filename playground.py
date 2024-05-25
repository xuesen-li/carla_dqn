class node:
    def __init__(self, val):
        self.val = val
    
    def __lt__(self, other):
        return self.val < other.val
    
    def __repr__(self):
        return str(self.val)
        
    

node1 = node(2)
node2 = node(1)
node3 = node(3)

nodes = [node1, node2, node3]
print(nodes)
nodes.sort()
print(nodes)