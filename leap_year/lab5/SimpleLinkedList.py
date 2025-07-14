from jmespath.ast import current_node
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


# class linkedListNode:
#     def __init__(self, value, nextNode=None):
#         self.value = value
#         self.nextNode = nextNode
#
# node1 = linkedListNode("1")
# node2 = linkedListNode("2")
# node3 = linkedListNode("3")
# node4 = linkedListNode("4")
# node5 = linkedListNode("5")
# node6 = linkedListNode("6")
# node7 = linkedListNode("7")
# node8 = linkedListNode("8")
#
# node1.nextNode = node2
# node2.nextNode = node3
# node3.nextNode = node4
# node4.nextNode = node5
# node5.nextNode = node6
# node6.nextNode = node7
# node7.nextNode = node8
#
# # current_node()
# currentNode = node1
#
# while True:
#     print(currentNode.value, ">>>", end=' ')
#
#     if currentNode.nextNode is None:
#         print("None")
#         break
#
#     currentNode = currentNode.nextNode
class LinkedList:
    def __init__(self):
        self.head =  None

    def insertAtBeginning(self, new_data):
        new_node = Node(new_data)
        new_node.next = self.head

        self.head = new_node

    def printList(self):
        temp = self.head

        while temp:
            print(f"{temp.data}",">>>", end=' ')
            temp =  temp.next

        print()

if __name__ == '__main__':
    llist = LinkedList()

    llist.insertAtBeginning("New")
    llist.insertAtBeginning("Data")
    llist.insertAtBeginning("Is")
    llist.insertAtBeginning("New")
    llist.insertAtBeginning("Data")
    llist.insertAtBeginning("Example")

    llist.printList()