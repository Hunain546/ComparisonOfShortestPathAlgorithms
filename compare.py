def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    
    # Check if files have the same number of lines
    if len(lines1) != len(lines2):
        print("The files have different number of lines.")
        return
    
    # Compare lines
    differences = []
    for i, (line1, line2) in enumerate(zip(lines1, lines2), start=1):
        if line1 != line2:
            differences.append((i, line1.strip(), line2.strip()))
    
    # Report results
    if differences:
        print("The files are different. Here are the differences:")
        for diff in differences:
            print(f"Line {diff[0]}: {file1} - '{diff[1]}' | {file2} - '{diff[2]}'")
    else:
        print("The files are the same.")

# Example usage:
compare_files("dijkstra_distances.txt", "bellman_distances.txt")
