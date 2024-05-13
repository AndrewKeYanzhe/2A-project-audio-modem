import csv
import numpy as np

# Open and read both CSV files
with open('channel.csv', 'r') as channel_file, open('file1.csv', 'r') as file1:
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

    for row in file1[:100]:
        print(row)




