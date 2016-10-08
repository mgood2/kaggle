import csv
with open('test_date.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    cnt = 0
    for row in spamreader:
        print ', '.join(row)
        cnt = cnt + 1
        if cnt == 2:
            break
