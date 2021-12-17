# marks the article lookup table based on if the article was published before or after the author received the
# excellence grant

from HelperObjects import *
from tqdm import tqdm
sql.connect()

sql.cur.execute('UPDATE lookup_article_researcher SET before=0')
sql.cur.execute('''SELECT DISTINCT(r.scopus_id) , e.grant_year FROM researchers r LEFT JOIN excellence_clusters e 
                    on e.cluster_id=r.cluster_id''')
results = sql.cur.fetchall()
for scopus_id, year in tqdm(results):
    sql.cur.execute('''UPDATE lookup_article_researcher SET before=1
                        WHERE researcher_id = ? 
                        AND article_id IN 
                        (SELECT a.eid FROM articles a INNER JOIN lookup_article_researcher l 
                            ON a.eid = l.article_id
                            AND l.researcher_id=?
                            AND a.year < ?)''', (scopus_id, scopus_id, year))

    sql.conn.commit()

