class Node:
    def __init__(self, data, next):
        self.data = data
        self.next = None
        
class SimpleLinkedList:
    def __init__(self):
        self.head = None

    def insetAtBegining(self, new_data):
        newNode = Node(new_data)
        newNode.next = self.head
        self.head = newNode
        
    def printList(self):
        ten = self.head
        while ten:
            print(f"{ten.data}", end=' ')
            ten = ten.Next
        print()
            
    if __name__ == "__main__":
        linkedList = SimpleLinkedList()
        linkedList.insetAtBegining("zero")	
        linkedList.insetAtBegining("one")
        linkedList.insetAtBegining("two")
        linkedList.insetAtBegining("three")
        linkedList.insetAtBegining("four")
        linkedList.insetAtBegining("five")
        linkedList.printList()










































# class SimpleLinkedList:
#     def __init__(self, value, nextNode=None):
#         self.value = value
#         self.nextNode = nextNode
# node1 = SimpleLinkedList("1")
# node2 = SimpleLinkedList("2")
# node3 = SimpleLinkedList("3")
# node4 = SimpleLinkedList("4")

# node1.nextNode = node2
# node2.nextNode = node3
# node3.nextNode = node4

# currentNode = node1

# while True:
#     print(currentNode.value, ">>>")

#     if currentNode.nextNode is None:
#         print("End of list")
#         break

#     currentNode = currentNode.nextNode