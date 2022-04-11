import sqlite3
from PyQt5 import QtWidgets, QtCore, QtGui
from datetime import datetime

class DatabaseManager():
    
    def open_db_registeredemployee(self):
        # print("THE ACCOUNT: "+ LOGIN_USER)
        # Create a database or connect to one
        conn = sqlite3.connect('facemaskdetectionDB.db')
        # Create a cursor
        c = conn.cursor()
        
        # Create table
        c.execute("""CREATE TABLE if not exists registeredemployee(
                id_number TEXT UNIQUE,
                first_name TEXT,
                last_name TEXT,
                status TEXT,
                registered_by TEXT
            )
            """) 

        # Commit changes
        conn.commit()
        # Close connection
        conn.close()

    def save_face(self, employee_id, face, added_by):
        try:
            self.labelError.setText('')
            # Create a database or connect to one
            conn = sqlite3.connect('facemaskdetectionDB.db')
            c = conn.cursor()
            # Insert user to the database
            if self.btnSave.text() == 'SAVE':
                c.execute(
                    "INSERT INTO registeredemployee VALUES(:employee_id, :face, :added_by)",
                    {
                        'employee_id': employee_id,
                        'face': face,
                        'added_by': added_by,
                    }
                    )
            elif self.btnSave.text() == 'UPDATE':
                c.execute(
                    "INSERT OR REPLACE INTO registeredemployee VALUES(:id_number, :first_name, :last_name, :status, :registered_by)",
                    {
                        'id_number': self.lineId.text(),
                        'first_name': self.lineFirstName.text(),
                        'last_name': self.lineLastName.text(),
                        'status': self.statusbtn.text(),
                        #'registered_by': LOGIN_USER,
                    }
                    )
            # Commit changes
            conn.commit()
            # Close connection
            conn.close()

            # Pop up message box
            msg = QMessageBox()
            msg.setWindowTitle('Saved to the Database!')
            msg.setText('User has been saved')
            msg.setIcon(QMessageBox.Information)
            x = msg.exec_()

            self.clearDetails()
        except sqlite3.Error as er:
            self.lineId.setStyleSheet(stylesheets.haserrorline)
            msg = QMessageBox()
            msg.setWindowTitle('ERROR!')
            msg.setText('Id number must be unique')
            msg.setIcon(QMessageBox.Critical)
            x = msg.exec_()
    
    def open_db_system_logs(self):
        # print("THE ACCOUNT: "+ LOGIN_USER)
        # Create a database or connect to one
        conn = sqlite3.connect('facemaskdetectionDB.db')
        # Create a cursor
        c = conn.cursor()
        
        # Create table
        c.execute("""CREATE TABLE if not exists system_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                action TEXT,
                user_id TEXT
            )
            """) 

        # Commit changes
        conn.commit()
        # Close connection
        conn.close()
        
        
    def open_db_detection_logs(self):
        # print("THE ACCOUNT: "+ LOGIN_USER)
        # Create a database or connect to one
        conn = sqlite3.connect('facemaskdetectionDB.db')
        # Create a cursor
        c = conn.cursor()
        
        # Create table
        c.execute("""CREATE TABLE if not exists detection_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                employee_id TEXT,
                user_id TEXT
            )
            """) 

        # Commit changes
        conn.commit()
        # Close connection
        conn.close()
    
    
        
class InsertDatabase():
    def insert_system_logs(self, action, user_id):
        # Create a database or connect to one
        conn = sqlite3.connect('facemaskdetectionDB.db')
        # Create a cursor
        c = conn.cursor()
        c.execute("INSERT INTO system_logs VALUES(null, :date, :action, :user_id)",
                {
                    'date': datetime.now().isoformat(' ', 'seconds'),
                    'action': action,
                    'user_id': user_id,      
                }
                )
        
        # Commit changes
        conn.commit()
        # Close connection
        conn.close()
        
        
    def insert_detection_logs(self, employee_id, user_id):
        # Create a database or connect to one
        conn = sqlite3.connect('facemaskdetectionDB.db')
        # Create a cursor
        c = conn.cursor()
        c.execute("INSERT INTO detection_logs VALUES(null, :date, :employee_id, :user_id)",
                {
                    'date': datetime.now().isoformat(' ', 'seconds'),
                    'employee_id': employee_id,
                    'user_id': user_id,      
                }
                )
        
        # Commit changes
        conn.commit()
        # Close connection
        conn.close()