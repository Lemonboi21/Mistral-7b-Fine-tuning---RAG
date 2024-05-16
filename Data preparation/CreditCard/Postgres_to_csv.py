
import psycopg2
import csv

# connect to the database
conn = psycopg2.connect(
    host="localhost",
    database="PFE",
    user="postgres",
    password="postgres")

print("-------------------------------------")
print("Connected to the database")
print("-------------------------------------")

# create a cursor
cur = conn.cursor()

# execute a statement
cur.execute("SELECT * FROM CreditCard")

# fetch the data
data = cur.fetchall()

# print the data
for row in data:
    print(row)

print("-------------------------------------")
print("Data fetched")
print("-------------------------------------")

# close the cursor
cur.close()

# close the connection
conn.close()

# write the data to a csv file

# add the header
data.insert(0, ['CreditCardID', 'Type', 'Number', 'ExpirationMonth', 'ExpirationYear', 'LastModified'])

with open('credit_card.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("-------------------------------------")
print("Data written to a csv file")
print("-------------------------------------")
















