import csv
import numpy as np

# Open and read both CSV files
with open('channel.csv', 'r') as channel_file, open('file1.csv', 'r') as file1, open('file2.csv', 'r') as file2, open('file3.csv', 'r') as file3, open('file4.csv', 'r') as file4, open('file5.csv', 'r') as file5, open('file6.csv', 'r') as file6:
    channel_reader = csv.reader(channel_file)
    file1_reader = csv.reader(file1)
    
    # print("Contents of channel.csv:")
    # for row in channel_reader:
        
    #     print(row)
    
    # print("\nContents of file1.csv:")
    # for row in file1_reader:
        
    #     print(row)

    channel=list(channel_reader)
    file1=list(file1_reader)

    print(len(file1))

    # for row in file1[:100]:
    #     print(row)




