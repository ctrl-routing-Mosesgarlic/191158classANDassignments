from node import *

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, data):
        node = Node(data)  # Create a new node, pass the value to be be appended
        if self.tail:
            self.tail.next = node
            self.tail = node
        else:
            # if there are no elements then the new node becomes the head and tail.
            self.head = node
            self.tail = node

        self.size += 1

    def append_at_a_location(self, data, index):
        # check for invalid input.
        if index < 0 or index > self.size:
            print("Invalid index")
            return

        node = Node(data)

        if index == 0:
            # Insert at beginning/start of the linked list
            node.next = self.head
            self.head = node
            if self.size == 0:
                self.tail = node
        else:
            current = self.head
            for i in range(index - 1):
                current = current.next
            node.next = current.next
            current.next = node
            if node.next is None:
                self.tail = node

        self.size += 1

    def display(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def search(self, data):
        current = self.head
        index = 0
        while current:
            if current.data == data:
                return index  # Return index where data is found
            current = current.next
            index += 1
        return -1  # Return -1 when data is not found

    def delete(self, data):
        current = self.head
        prev = None

        while current:
            if current.data == data:
                if prev:
                    prev.next = current.next
                else:
                    self.head = current.next

                if current.next is None:
                    self.tail = prev

                self.size -= 1
                return True  # if found
            prev = current
            current = current.next

        return False  # if not found return false

    def clear(self):
        # clear all elements from the linkedlist,
        self.tail = None
        self.head = None

