import pandas as pd
import psycopg2
import csv
import os

# connect to the database

conn = psycopg2.connect(
    host="localhost",
    database="PFE",
    user="postgres",
    password="postgres"
)

print("Database opened successfully")

# create a cursor
cur = conn.cursor()

# create the sql query
sql_query = "SELECT * FROM CreditCard"

# execute the sql query
cur.execute(sql_query)

# fetch the data
data = cur.fetchall()


#--------------------------------- create the csv files ---------------------------------#

#---------------------------------------------------------
# get the credit card informations using the credit card id
#---------------------------------------------------------

def prompt_response_generator(data, csv_file_path):

    csv_file_path = 'csv/CreditCardID.csv'
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Prompt", "Response"])

        for row in data:
            prompt = "What are the credit card informations of the card with the id of " + str(row[0]) + "?"
            response = "The credit card with the id of " + str(row[0]) + " is a " + row[1] + " with the number of " + \
                       row[2] + " and the expiration date of " + str(row[3]) + "/" + str(row[4]) + " and it was last modified on " + \
                       str(row[5])
            writer.writerow([prompt, response])

prompt_response_generator(data, 'csv/CreditCardID.csv')





#---------------------------------------------------------
# create a csv file with only the responses
#---------------------------------------------------------

def responses_generator(data, csv_file_path):

    csv_file_path = 'csv/CreditCardID_responses.csv'
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Response"])

        for row in data:
            response = "The credit card with the id of " + str(row[0]) + " is a " + row[1] + " with the number of " + row[2] + " and the expiration date of " + str(row[3]) + "/" + str(row[4]) + " and it was last modified on " + str(row[5])
            writer.writerow([response])





# close the connection
conn.close()
print("Connection closed successfully")
