'''Main module for running application'''

from modules.GUI import GUI
from modules.database import Database


def main():
    '''Main function'''
    db = Database() #create database
    GUI(db) #start app

if __name__ == '__main__':
    main()
