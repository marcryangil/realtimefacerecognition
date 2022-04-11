import os
import sqlite3

def start():
    savedfacesid = list()

    try:
        connection = sqlite3.connect("facemaskdetectionDB.db")
        cur = connection.cursor()
        sqlquery = "SELECT id FROM RegisteredFaces"

        for row in cur.execute(sqlquery):
            savedfacesid.append(str(row[0]))
        connection.commit()
        connection.close()
    except Exception as e:
        print(e)

    ref_dir = r'C:\UFMDSdatabase'

    for dirname, subdirname, filenames in os.walk(ref_dir):
        if len(filenames) > 0:
            for filename in filenames:
                file = filename.split(".")[0]
                if file not in savedfacesid:
                    file_path = os.path.join(dirname, filename)
                    print(file_path)
                    os.remove(file_path)