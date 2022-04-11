import base64
import os
import sqlite3


def save():
    try:
        connection = sqlite3.connect("facemaskdetectionDB.db")
        cur = connection.cursor()
        sqlquery = "SELECT employee_id," \
                   "(SELECT first_name FROM registeredemployee WHERE id_number = RegisteredFaces.employee_id) AS fname," \
                   "(SELECT last_name FROM registeredemployee WHERE id_number = RegisteredFaces.employee_id) AS lname , " \
                   "face," \
                   "id " \
                   "FROM RegisteredFaces"

        for row in cur.execute(sqlquery):
            directory = row[0] + '-' + row[1] + '-' + row[2]
            parent_dir = r"C:\UFMDSdatabase"
            path = os.path.join(parent_dir, directory)

            if os.path.exists(path) == False:
                os.makedirs(path)

            filename = r"\{}.jpg".format(row[4])

            with open(path+filename, "wb") as fh:
                fh.write(base64.b64decode(row[3]))
        connection.commit()
        connection.close()
    except Exception as e:
        print(e)



