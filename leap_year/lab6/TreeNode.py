kuna hii class TreeNode:
    def _init_(self,value):
        self.right=None
        self.left=None
        self.value=value

    def insert(self,key_value):
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
        print(self.value)

        if self.left:
            self.left.preorder_traversal()
        
        if self.right:
            self.right.preorder_traversal()

    def inorder_traversal(self):
        if self.left:
            self.left.inorder_traversal()

        print(self.value)

        if self.right:
            self.right.inorder_traversal()

    def postorder_traversal(self):
        if self.left:
            self.left.postorder_traversal()

        if self.right:
            self.right.postorder_traversal()

        print(self.value)

if _name_ == '_main_':
    tree_obj = TreeNode(17)
    tree_obj.insert(5)
    tree_obj.insert(4)
    tree_obj.insert(6)

    tree_obj.insert(23)
    tree_obj.insert(20)
    tree_obj.insert(7)

    print("\n Preorder traversal")
    tree_obj.preorder_traversal()
    
    print("\n Inorder traversal")
    tree_obj.inorder_traversal()
    
    print("\n Postorder traversal")
    tree_obj.postorder_traversal()