# compare_line_lists.py
def read_file_to_set(filename):
    with open(filename, 'r') as file:
        return set(line.strip() for line in file if line.strip())

def write_differences_to_file(differences, output_filename):
    with open(output_filename, 'w') as file:
        file.write('\n'.join(sorted(differences)))

def main():
    file1 = '/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/code/subjects.txt'
    file2 = '/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/code/subjects_T1.txt'
    output_file = '/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/code/subjects_paired_T1.txt'

    set1 = read_file_to_set(file1)
    set2 = read_file_to_set(file2)

    differences = set1.symmetric_difference(set2)  # Items in either set1 or set2 but not both

    write_differences_to_file(differences, output_file)
    print(f"Differences written to {output_file}")

if __name__ == "__main__":
    main()
