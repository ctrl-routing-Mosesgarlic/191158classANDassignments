class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    def insert(self, key_value):
        if key_value < self.value:
            
            if self.left is None:
                self.left = TreeNode(key_value)
            else:
                self.left.insert(key_value)
        
        else:
            if self.right is None:
                self.right = TreeNode(key_value)
            else:
                self.right.insert(key_value)
    def preorder_traversal(self):
        print(self.value):
            
        if self.left:
            self.left.preorder_traversal()
        if self.right:
            self.right.preorder_traversal()
    def postorder_traversal(self):
        if self.left:
            self.left.postorder_traversal()
        if self.right:
            self.right.postorder_traversal()
        print(self.value)
        
        
    if __name__ == "__main__":
        root = TreeNode(10)
        root.insert(5)
        root.insert(15)
        root.insert(3)
        root.insert(7)
        root.insert(12)
        root.insert(18)

        print("Preorder Traversal:")
        root.preorder_traversal()

        print("\nPostorder Traversal:")
        root.postorder_traversal()