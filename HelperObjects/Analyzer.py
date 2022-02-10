from tqdm import tqdm
from HelperObjects import *
import numpy as np
import pandas as pd
import gensim


class Analyzer:
    def __init__(self, model):
        self.name = model.upper()
        self.sql = sql
        self.sql.connect()
        self.paths = pathManager.get_paths()
        self.df = pd.read_pickle(self.paths['df_parsed_articles'])
        self.dict = self.df.set_index('id').to_dict('index')
        self.model = gensim.models.LdaModel.load(f'DATA/MODELS/MODEL/{self.name}/best_model.model')

    # get the topic distribution of articles of a author, and divide the articles into before/after groups based on the
    # cluster group given into it
    def get_article_topics_before_after(self, auth_id, cluster_id):
        self.sql.cur.execute('SELECT DISTINCT(grant_year) from excellence_clusters WHERE cluster_id=?', (cluster_id,))
        year = self.sql.cur.fetchall()[0][0]
        self.sql.cur.execute('''SELECT DISTINCT a.eid, a.topic_position FROM articles a INNER JOIN lookup_article_researcher l 
                                    ON a.eid = l.article_id
                                    AND l.researcher_id=?
                                    AND a.year < ?
                                    AND a.abstract_flag=1''', (auth_id, year))
        temp = self.sql.cur.fetchall()
        before = [([i[1] for i in self.dict[eid]['topics']]) for eid, _ in temp]

        self.sql.cur.execute('''SELECT DISTINCT a.eid, a.topic_position FROM articles a INNER JOIN lookup_article_researcher l 
                                    ON a.eid = l.article_id
                                    AND l.researcher_id=?
                                    AND a.year >= ?
                                    AND a.abstract_flag=1''', (auth_id, year))
        temp = self.sql.cur.fetchall()
        after = [([i[1] for i in self.dict[eid]['topics']]) for eid, _ in temp]
        return before, after

    # returns the distance between two points
    @staticmethod
    def dist(x1, x2):
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
            dists.append((temp / (n - 1)))  # n-1 for when i=j
        return dists

    # get the cluster id and author pairs for all authors
    def get_grants(self, first=True):
        # divides the authors works into 2 based on the first grant they ever received, disregards the others
        if first:
            self.sql.cur.execute(
                'SELECT scopus_id, MIN(cluster_id) FROM researchers GROUP BY scopus_id HAVING scopus_id NOT NULL')
            return self.sql.cur.fetchall()

        # divides the authors works into multiple groups of before and after, for every grant they ever recieved
        # e.g.: an author recieves two grants, one in 2000, a second one in 2010
        #       this method gives 2 entries for that author, one as before/after 2000, second as before/after 2010
        self.sql.cur.execute("SELECT scopus_id, cluster_id FROM researchers WHERE scopus_id NOT NULL")
        return self.sql.cur.fetchall()

    # add the topic distribution of all the documents
    def calc_position_of_documents(self):
        print('Calculating results')
        self.df['topics'] = self.df['bow'].map(lambda x: self.model.get_document_topics(x, minimum_probability=0))
        self.dict = self.df.set_index('id').to_dict('index')

    def calc_before_after_grant_average_dist(self):
        self.sql.conn.commit()

        results = {
            'Cluster_id': [],
            'Author_id': [],
            'size_before': [],
            'avg_dist_before': [],
            'size_after': [],
            'avg_dist_after': []
        }
        res = self.get_grants(first=True)

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

        results_df = pd.DataFrame(results)

        temp = results_df.drop(results_df[(results_df.avg_dist_after < 0.0) & (results_df.avg_dist_before < 0.0)].index)
        print(temp.avg_dist_before.mean(), '-', temp.avg_dist_after.mean())

        results_df.to_csv(self.paths['df_distance_csv'], sep='\t')
        results_df.to_pickle(self.paths['df_distance'])
        return results_df

    def calc_average_for_every_year(self):
        path = f"DATA/ANALYSIS/{self.name}_yearly_average_distances.pickle"
        try:
            return pd.read_pickle(path)
        except:
            self.calc_position_of_documents()
            auths = self.get_grants()
            auth_collector = {
            }
            for auth_id, _ in tqdm(auths):
                auth_collector[auth_id] = dict()
                self.sql.cur.execute('''SELECT DISTINCT a.year FROM articles a INNER JOIN lookup_article_researcher l 
                                                        ON a.eid = l.article_id
                                                        AND l.researcher_id=?
                                                        AND a.abstract_flag=1''', (auth_id, ))
                years = self.sql.cur.fetchall()
                for year_count, year in enumerate(years):
                    self.sql.cur.execute('''SELECT DISTINCT a.eid FROM articles a INNER JOIN lookup_article_researcher l 
                                                ON a.eid = l.article_id
                                                AND l.researcher_id=?
                                                AND a.year = ?
                                                AND a.abstract_flag=1''', (auth_id, int(year[0])))
                    temp = self.sql.cur.fetchall()
                    year_document_topics = [([i[1] for i in self.dict[eid[0]]['topics']]) for eid in temp]
                    avg_dist = self.map_to_average_distance(year_document_topics)

                    auth_collector[auth_id][year_count] = np.average(avg_dist)
            df = pd.DataFrame(auth_collector)
            df['average'] = df.mean(axis=1)
            df.sort_index(axis=0, inplace=True)
            df.to_pickle(path)
            return df

    def main(self):
        path = f"DATA/ANALYSIS/{self.name}_results.pickle"
        try:
            return pd.read_pickle(path)
        except:
            self.calc_position_of_documents()
            data = self.calc_before_after_grant_average_dist()
            data.set_index(data["Author_id"], inplace=True)
            data.drop(columns=["Author_id"], inplace=True)
            pd.to_pickle(data, path)
            return data
