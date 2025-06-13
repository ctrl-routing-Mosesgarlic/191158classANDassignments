from doubly_linked_list import DoublyLinkedList

if __name__ == '__main__':
    dll = DoublyLinkedList()
    dll.append(10)
    dll.append(20)
    dll.append(30)
    dll.display_forward()

    dll.prepend(5)
    dll.display_forward()

    dll.insert_at_a_specific_index(15, 2)
    dll.display_forward()

    print("Search for 15:", dll.search(15))

    dll.delete_at(2)
    dll.display_forward()

    print("Display backward:")
    dll.display_backward()
