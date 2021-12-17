import pybliometrics
from pybliometrics.scopus import AuthorRetrieval
from tqdm import tqdm
from HelperObjects import *



"""
This class uses the scopus API to download the Article ids that are affiliated with the author ids in the DATABANK 
"""
class ArticleIdDownloader:
    def __init__(self, progess_bar = True):
        self.conn, self.cur = sql.connect()
        self.progess_bar = progess_bar

    def main(self):
        self.cur.execute('SELECT scopus_id FROM researchers WHERE scopus_id NOTNULL')
        auth_ids = self.cur.fetchall()
        self.download_article_ids(auth_ids)

        # go again for the authors whose new ids are known
        self.cur.execute('SELECT alternative_scopus_id FROM researchers WHERE alternative_scopus_id NOTNULL')
        auth_ids = self.cur.fetchall()
        self.download_article_ids(auth_ids)

    def download_article_ids(self, auth_ids):
        missing_error_count = 0
        if self.progess_bar: iterable = tqdm(auth_ids)
        else: iterable = auth_ids
        for auth_id in iterable:
            auth_id = auth_id[0]
            response = AuthorRetrieval(auth_id)

            # The attribute error_no

            # The new id is known
            if response.error_no == 1:
                self.cur.execute("UPDATE researchers SET error=1, alternative_scopus_id=? WHERE scopus_id =  ?",
                            (response._id, auth_id))
                continue

            # the merged and ... one of these
            if response.error_no == 2:
                self.cur.execute("UPDATE researchers SET error=2 WHERE scopus_id =  ?", (auth_id,))
                continue

            articles = response.get_documents()
            # if there are no articles for that author
            if articles is None: continue

            for i in articles:
                eid = i.eid
                doi = i.doi
                pii = i.pii

                title = i.title
                author_ids = i.author_ids
                date = i.coverDate
                year = int(date[:4])

                source_id = i.source_id
                volume = i.volume
                issue = i.issn
                article_no = i.article_number

                page_range = i.pageRange

                cited_by_count = i.citedby_count
                link = None

                afil_id = i.afid

                author_afids = i.author_afids

                author_keywords = i.authkeywords
                document_type = i.subtypeDescription

                # save the link
                self.cur.execute('INSERT INTO lookup_article_researcher (researcher_id, article_id) VALUES (?,?)',
                            (auth_id, eid))

                # save the document
                self.cur.execute(
                    ''' 
                    INSERT INTO articles (eid, doi, pii, title, author_ids, year, source_id, volume, issue, article_no, cited_by_count, affiliations, authors_with_affiliations, author_keywords, document_type, page_start, date, abstract_flag) 
                    VALUES               (?,   ?,   ?,   ?,     ?,          ?,    ?,         ?,      ?,     ?,          ?,              ?,            ?,                         ?,               ?            , ?        ? ,0)
                    ''', (eid, doi, pii, title, author_ids, year, source_id, volume, issue, article_no, cited_by_count,
                          afil_id, author_afids, author_keywords, document_type, page_range, date)
                )

                # record the affiliation in its own table

                # some articles do not have any affiliations recorded, so for them this part can be skipped
                if i.afid is None: continue

                # parse and save the affiliations
                afil_ids = i.afid.split(';')
                affil_names = i.affilname.split(';')
                affil_citys = i.affiliation_city.split(';')
                affil_countrys = i.affiliation_country.split(';')

                '''
                There are some cases where the count is not the same for all of them, so trying to parse them casuse an index error.
                I assumed that there were some faults in the scopus database. I trust the affiliation id above all other entries,
                so if the count is not the same we disregard it completely to take a look at the faulty ones later.

                We can not be sure which one of the ids have a missing entry, because the spi only returns a list, and we do not
                which element was missing.
                '''
                af_count = len(afil_ids)

                # we can only assume that the affiliation ids were correct and the other entries have been mistaken, so if there
                # are any inconsistencies with the return value, just disregard it, can be dealt with later with another script
                if af_count != len(affil_names):
                    affil_names = ['' for c in range(af_count)]
                    missing_error_count += 1
                if af_count != len(affil_citys):
                    affil_citys = ['' for c in range(af_count)]
                    missing_error_count += 1
                if af_count != len(affil_countrys):
                    affil_countrys = ['' for c in range(af_count)]
                    missing_error_count += 1

                for a in range(af_count):
                    af_id = afil_ids[a]
                    af_name = affil_names[a]
                    af_city = affil_citys[a]
                    af_country = affil_countrys[a]
                    self.cur.execute(
                        '''
                        INSERT INTO affiliations (id,    name,    city,    country   ) VALUES (?,?,?,?)
                        ''', (af_id, af_name, af_city, af_country)
                    )

                # save the connection
            # self.conn.commit()
