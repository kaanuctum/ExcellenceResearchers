from pybliometrics.scopus import AbstractRetrieval
from tqdm import tqdm
import threading
import queue

from HelperObjects import *

class SingleThreadedDownloader:
    def __init__(self):
        self.conn, self.cur = sql.connect()
        self.cur.execute('SELECT DISTINCT(eid) FROM articles where abstract_flag=0')
        self.article_ids = [i[0] for i in self.cur.fetchall()]

    def main(self):
        for eid in tqdm(self.article_ids):
            try:
                res = AbstractRetrieval(eid, view='FULL')
                ab = res.abstract
                dsc = res.description
                sbj = res.subject_areas
            except Exception as e:
                # self.display.display(msg=str(e))
                if str(e) == "The resource specified cannot be found.":
                    self.cur.execute("UPDATE articles SET abstract_flag=-1 WHERE eid=?", (eid,))
                    continue
                elif str(e) == "Quota Exceeded": exit()  # we don't want to upset the good people at Elsevier
                else: continue
            if ab is None:  # article couldn't be retrieved
                continue
            self.cur.execute("INSERT INTO article_abstracts (article_id, abstract, description) VALUES (?, ?, ?)",
                             (eid, ab, dsc))
            self.cur.execute("UPDATE articles SET abstract_flag=1 WHERE eid=?", (eid,))

            # some articles do not specify their subjects
            if sbj is None: continue
            for subject in sbj:
                self.cur.execute("INSERT INTO subjects (code, area, abbreviation) VALUES (?, ?, ?)",
                                 (subject[2], subject[0], subject[1]))
                self.cur.execute("INSERT INTO lookup_article_subject (article_id, subject_id) VALUES (?, ?)",
                                 (eid, subject[2]))



class QueuedThreadedDownloader:
    def __init__(self):
        self.thread_count = 5  # cant go up too much, the Elsevier servers get upset
        self.queue = queue.Queue()
        self.threads = list()

        self.conn, self.cur = sql.connect()
        self.cur.execute('SELECT eid FROM articles WHERE abstract_flag=0')
        self.article_ids = [i[0] for i in self.cur.fetchall()]

        # lets us see how many we got so far
        self.display = tqdm(total=len(self.article_ids))

    def articleDownloader(self, batch):
        for eid in batch:
            try:
                res = AbstractRetrieval(eid, view='FULL')
                ab = res.abstract
                dsc = res.description
                sbj = res.subject_areas
                self.queue.put((eid, ab, dsc, sbj))
            except Exception as e:
                self.display.display(msg=str(e))
                if str(e) == "The resource specified cannot be found.": self.queue.put((eid, None, None, None))
                elif str(e) == "Quota Exceeded": exit()  # we don't want to upset the good people at Elsevier
                else: continue

    def articleWriter(self):
        while self.has_active_downloader():
            while not self.queue.empty():
                (eid, ab, dsc, sbj) = self.queue.get()
                if ab is None: # article couldn't be retrieved
                    self.cur.execute("UPDATE articles SET abstract_flag=-1 WHERE eid=?", (eid,))
                    continue
                self.cur.execute("INSERT INTO article_abstracts (article_id, abstract, description) VALUES (?, ?, ?)", (eid, ab, dsc))
                self.cur.execute("UPDATE articles SET abstract_flag=1 WHERE eid=?", (eid,))

                # some articles do not specify their subjects
                if sbj is None: continue
                for subject in sbj:
                    self.cur.execute("INSERT INTO subjects (code, area, abbreviation) VALUES (?, ?, ?)", (subject[2], subject[0], subject[1]))
                    self.cur.execute("INSERT INTO lookup_article_subject (article_id, subject_id) VALUES (?, ?)", (eid, subject[2]))
                self.display.update(1)
                self.conn.commit()

    def articleWriter_simple(self):
        while self.has_active_downloader():
            while not self.queue.empty():
                (eid, ab, dsc, sbj) = self.queue.get()
                if ab is None: # article couldn't be retrieved
                    self.cur.execute("UPDATE articles SET abstract_flag=-1 WHERE eid=?", (eid,))
                    continue
                self.cur.execute("INSERT INTO article_abstracts (article_id, abstract, description) VALUES (?, ?, ?)",(eid, ab, dsc))
                self.cur.execute("UPDATE articles SET abstract_flag=1 WHERE eid=?", (eid,))
                self.display.update(1)


    def has_active_downloader(self):
        for t in self.threads:
            if t.is_alive(): return True
        return False

    def main(self):
        for i in range(self.thread_count):
            batch_size = int(len(self.article_ids) / self.thread_count) + 1
            batch = self.article_ids[i * batch_size: (i + 1) * batch_size]
            x = threading.Thread(target=self.articleDownloader, args=(batch,), daemon=True)
            self.threads.append(x)
            x.start()
        self.articleWriter()
        for t in self.threads:
            t.join()


class DownloaderFactory:
    def __init__(self, t):
        if t == "queue":
            self.downloader = QueuedThreadedDownloader()
        elif t == "single":
            self.downloader = SingleThreadedDownloader()
        else:
            raise RuntimeError("Not a valid ype of downloader")

    def main(self):
        self.downloader.main()


if __name__ == "__main__":
    downloader = DownloaderFactory("queue")
    downloader.main()
