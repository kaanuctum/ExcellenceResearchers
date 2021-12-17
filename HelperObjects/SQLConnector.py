import sqlite3
import pyodbc
from tqdm import tqdm

class SQLConnector:
    def __init__(self, lite_name=None, name=None, pswd=None):
        self.sqllite_db_name = lite_name
        self.server_name = name
        self.server_pswd = pswd
        self.conn = None
        self.cur = None

    def connect(self):
        # not happy with the implmentation
        # for now it assumes, if the sqllite name is set, that we use the sqlite version
        if self.conn is None or self.cur is None:
            if self.sqllite_db_name is not None:
                self.conn, self.cur = self.connect_to_sql_sqlite()
            else:
                self.conn, self.cur = self.connec_to_sql_server()
            self.cur.execute("BEGIN TRANSACTION")
        return self.conn, self.cur

    def connect_to_sql_sqlite(self):
        """
        a method to connect to the sqlite file
        the database name is hard coded
        If in the future we decide to connect it to a database server, the passwords and usernames can be seperated
        :return: the cursor and connection objects
        """
        if self.sqllite_db_name is None: raise RuntimeError("SQLlite db name is not set")
        conn = sqlite3.connect(self.sqllite_db_name, check_same_thread=False)
        cur = conn.cursor()
        return conn, cur

    def connec_to_sql_server(self):
        conn = pyodbc.connect('Driver={SQL Server};'
                              'Server=LITTLEBOY;'
                              'Database=Excellence;'
                              'Trusted_Connection=yes;')

        cur = conn.cursor()
        return conn, cur

    def delete_duplicate_articles(self):
        self.cur.execute("DELETE FROM articles WHERE ROWID NOT IN (SELECT min(rowid) FROM articles GROUP BY eid);")

    def delete_duplicate_subjects(self):
        self.cur.execute("DELETE FROM subjects WHERE ROWID NOT IN (SELECT min(ROWID) FROM subjects GROUP BY code);")

    def delete_duplicate_affiliations(self):
        self.cur.execute("DELETE FROM affiliations WHERE ROWID NOT IN (SELECT min(rowid) FROM affiliations GROUP BY id);")

    def update_cleaned_abstracts(self, input):
        print('updating cleaned abstracts')
        for (eid, cleaned) in tqdm(input):
            self.cur.execute("UPDATE article_abstracts SET cleaned_abstracts=? WHERE article_id=?", (cleaned, eid))
            if cleaned.find("©") > 0:
                self.cur.execute("UPDATE article_abstracts SET mark=2 WHERE article_id=?", (eid,))

    def nuke(self):
        self.cur.execute("DELETE FROM articles")
        self.cur.execute("DELETE FROM researchers")
        self.cur.execute("DELETE FROM affiliations")
        self.cur.execute("DELETE FROM lookup_article_researcher")
        self.cur.execute("DELETE FROM lookup_article_subject")
        self.cur.execute("DELETE FROM article_abstracts")
        self.cur.execute("DELETE FROM subjects")
        self.conn.commit()
        print("nuked")

    def create_tables(self):
        self.cur.execute('''
                BEGIN TRANSACTION;
        DROP TABLE IF EXISTS "lookup_article_subject";
        CREATE TABLE IF NOT EXISTS "lookup_article_subject" (
            "article_id"	INTEGER,
            "subject_id"	INTEGER
        );
        DROP TABLE IF EXISTS "subjects";
        CREATE TABLE IF NOT EXISTS "subjects" (
            "code"	INTEGER NOT NULL,
            "area"	TEXT NOT NULL,
            "abbreviation"	TEXT NOT NULL
        );
        DROP TABLE IF EXISTS "article_abstracts";
        CREATE TABLE IF NOT EXISTS "article_abstracts" (
            "article_id"	INTEGER NOT NULL UNIQUE,
            "abstract"	TEXT,
            "description"	TEXT
        );
        DROP TABLE IF EXISTS "articles";
        CREATE TABLE IF NOT EXISTS "articles" (
            "eid"	INTEGER,
            "doi"	TEXT,
            "pii"	TEXT,
            "title"	TEXT,
            "author_ids"	TEXT,
            "year"	INTEGER,
            "source_id"	TEXT,
            "volume"	INTEGER,
            "issue"	INTEGER,
            "article_no"	INTEGER,
            "page_start"	INTEGER,
            "page_end"	INTEGER,
            "page_range"	INTEGER,
            "cited_by_count"	INTEGER,
            "link"	TEXT,
            "affiliations"	TEXT,
            "authors_with_affiliations"	TEXT,
            "author_keywords"	TEXT,
            "document_type"	TEXT,
            "source"	TEXT,
            "abstract_flag"	INTEGER
        );
        DROP TABLE IF EXISTS "researchers";
        CREATE TABLE IF NOT EXISTS "researchers" (
            "internal_id"	INTEGER NOT NULL,
            "name"	TEXT NOT NULL,
            "scopus_id"	INTEGER,
            "cluster_id"	INTEGER,
            "alternative_scopus_id"	INTEGER,
            "error"	INTEGER
        );
        DROP TABLE IF EXISTS "lookup_article_researcher";
        CREATE TABLE IF NOT EXISTS "lookup_article_researcher" (
            "researcher_id"	INTEGER NOT NULL,
            "article_id"	INTEGER NOT NULL
        );
        DROP TABLE IF EXISTS "affiliations";
        CREATE TABLE IF NOT EXISTS "affiliations" (
            "id"	INTEGER,
            "name"	TEXT,
            "city"	TEXT,
            "country"	TEXT
        );
        COMMIT;

        ''')

# create the object, that would be called upon
