from tqdm import tqdm
from HelperObjects import *
import numpy as np
import pandas as pd
import gensim


class Analyzer:
    def __init__(self):
        self.sql = sql
        self.sql.connect()
        self.paths = pathManager.get_paths()
        self.df = pd.read_pickle(self.paths['df_parsed_articles'])
        self.dict = self.df.set_index('id').to_dict('index')
        self.model = gensim.models.LdaModel.load(self.paths['model'])
        self.results_df = pd.DataFrame

    def get_article_topics_before_after(self, auth_id, cluster_id):
        self.sql.cur.execute('SELECT DISTINCT(grant_year) from excellence_clusters WHERE cluster_id=?', (cluster_id,))
        year = self.sql.cur.fetchall()[0][0]
        self.sql.cur.execute('''SELECT a.eid, a.topic_position FROM articles a INNER JOIN lookup_article_researcher l 
                                    ON a.eid = l.article_id
                                    AND l.researcher_id=?
                                    AND a.year < ?
                                    AND a.abstract_flag=1''', (auth_id, year))
        temp = self.sql.cur.fetchall()
        before = [([i[1] for i in self.dict[eid]['topics']]) for eid, _ in temp]

        self.sql.cur.execute('''SELECT a.eid, a.topic_position FROM articles a INNER JOIN lookup_article_researcher l 
                                    ON a.eid = l.article_id
                                    AND l.researcher_id=?
                                    AND a.year >= ?
                                    AND a.abstract_flag=1''', (auth_id, year))
        temp = self.sql.cur.fetchall()
        after = [([i[1] for i in self.dict[eid]['topics']]) for eid, _ in temp]
        return before, after

    # returns the distance between two points
    def dist(self, x1, x2):
        a = np.array(x1)
        b = np.array(x2)
        if a.shape != b.shape: raise RuntimeError('The two inputs do not have the same dimensions')
        return np.sqrt(((a - b) ** 2).sum())

    # takes in a list of inputs, and maps the values in the list to the average distance of the point to all the other
    # points
    def map_to_average_distance(self, inputs):
        n = len(inputs)
        dists = []
        for i in range(n):
            temp = 0.0
            for j in range(n):
                temp += self.dist(inputs[i], inputs[j])
            dists.append((temp / (n - 1)))
        return dists

    # divides the authors works into 2 based on the first grant they ever recieved
    def get_first_grants(self):
        self.sql.cur.execute(
            'SELECT scopus_id, MIN(cluster_id) FROM researchers GROUP BY scopus_id HAVING scopus_id NOT NULL')
        return self.sql.cur.fetchall()

    # divides the authors works into multiple groups of before and after, for every grant they ever recieved
    # e.g.: an author recieves two grants, one in 2000, a second one in 2010
    #       this method gives 2 entries for that author, one as before/after 2000, second as before/after 2010
    def get_all_grants(self):
        self.sql.cur.execute("SELECT scopus_id, cluster_id FROM researchers")
        return self.sql.cur.fetchall()

    def calc_before_after_average_dist(self):
        self.sql.cur.execute("update researchers SET avg_dist_before=0, avg_dist_after=0")
        self.sql.conn.commit()

        results = {
            'Cluster_id': [],
            'Author_id': [],
            'size_before': [],
            'avg_dist_before': [],
            'size_after': [],
            'avg_dist_after': []
        }
        # res = self.get_all_grants()
        res = self.get_first_grants()

        for auth_id, cluster_id in tqdm(res):
            before, after = self.get_article_topics_before_after(auth_id, cluster_id)

            # basic checks for all the 'wrong' inputs we can have

            if len(before) == 0:  # no input before
                avg_dist_before = -1.0
            elif len(before) == 1:  # only 1 input so the 'distance' between topics does not make sense
                avg_dist_before = -2.0
            else:
                avg_dist_before = np.array(self.map_to_average_distance(before)).mean()

            if len(after) == 0:  # no input before
                avg_dist_after = -1.0
            elif len(after) == 1:  # only 1 input so the 'distance' between topics does not make sense
                avg_dist_after = -2.0
            else:
                avg_dist_after = np.array(self.map_to_average_distance(after)).mean()

            # save results
            results["Cluster_id"].append(cluster_id)
            results["Author_id"].append(auth_id)
            results["size_before"].append(len(before))
            results["avg_dist_before"].append(avg_dist_before)
            results["size_after"].append(len(after))
            results["avg_dist_after"].append(avg_dist_after)

        self.results_df = pd.DataFrame(results)

        temp = self.results_df.drop(self.results_df[(self.results_df.avg_dist_after < 0.0) & (self.results_df.avg_dist_before < 0.0)].index)
        print(temp.avg_dist_before.mean(), '-', temp.avg_dist_after.mean())

        self.results_df.to_csv(self.paths['df_distance_csv'], sep='\t')
        self.results_df.to_pickle(self.paths['df_distance'])

    def calc_position_of_documents(self):
        print('Calculating results')
        self.df['topics'] = self.df['bow'].map(lambda x: self.model.get_document_topics(x, minimum_probability=0))
        self.dict = self.df.set_index('id').to_dict('index')
        self.df.to_pickle(self.paths['df_parsed_articles'])
        self.df.to_csv('DATA/prepared_data.csv')

    def write_distances_to_db(self):
        for row in self.results_df.iterrows():
            self.sql.cur.execute(
                "update researchers SET avg_dist_before=?, avg_dist_after=? WHERE scopus_id=? AND cluster_id=?",
                (row['avg_dist_before'], row['avg_dist_after'], row['auth_id'], row['cluster_id']))
            self.sql.conn.commit()

    def write_position_of_documents(self):
        print('writing results into db')
        try:
            for i, row in tqdm(self.df.iterrows()):
                sql.cur.execute("UPDATE articles SET topic=? WHERE eid=?", (row['topics'], row['id']))
                sql.conn.commit()
        except:
            self.calc_position_of_documents()
            for i, row in tqdm(self.df.iterrows()):
                sql.cur.execute("UPDATE articles SET topic=? WHERE eid=?", (row['topics'], row['id']))
                sql.conn.commit()