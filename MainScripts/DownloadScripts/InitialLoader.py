"""
Script to move the names and ids of authors we are intereseted in, into the .db file

No longer needed as the data was already exported into the db once and the excel file was deleted afterwards
TODO: delete the script afterwards, or move it to old scripts to keep better track of the progress
"""
from HelperObjects import *
import pandas as pd

def load_researchers():
    # read the csv file first
    df = pd.read_excel(dataFhand["excel_name"])

    # connect to the database
    conn, cur = sql.connect()

    # add in the elements

    for i in df.index:
        cluster_id = df['Cluster ID'][i]
        name = df['Name'][i].strip()
        scopus_id = df['Scopus ID'][i]
        err = 1
        if pd.isnull(df['Altnative ID'][i]) and pd.isnull(df['Remarks'][i]): err = 0
        cur.execute('INSERT INTO researchers (internal_id, name, scopus_id, cluster_id, error) VALUES (?,?,?,?,?) ',
                    (i, name, scopus_id, int(cluster_id), err))

    conn.commit()


def load_clusters():
    df = pd.read_excel(dataFhand["excel_name"], sheet_name='institutions')
    sql.connect()
    sql.cur.execute('DELETE FROM excellence_clusters')
    for i, row in df.iterrows():
        sql.cur.execute('INSERT INTO excellence_clusters (cluster_id, grant_year, name, subject) VALUES (?,?,?,?)',
                        (row['Cluster ID'], row['Start'], row["Name"], row["Subject Area"]))

    sql.conn.commit()

