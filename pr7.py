def tower_of_hanoi(n, source, target, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    tower_of_hanoi(n-1, source, auxiliary, target)
    print(f"Move disk {n} from {source} to {target}")
    tower_of_hanoi(n-1, auxiliary, target, source)

try:
    n = int(input("Enter number of disks: "))
    if n <= 0:
        print("Please enter a positive number of disks.")
    else:
        tower_of_hanoi(n, 'A', 'C', 'B')
except ValueError:
    print("Invalid input. Please enter a number.")