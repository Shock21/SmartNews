from __future__ import print_function

import newspaper
import sys

from pyspark.sql import SparkSession
from newspaper import Article
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg.distributed import BlockMatrix

#from sklearn.feature_extraction.text import TfidfVectorizer

def getArticletText(urls):
    article = Article(url=urls, language='en', fetch_images=False)
    article.download()
    article.parse()
    return article.text

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: news <file>", file=sys.stderr)
        exit(-1)

    spark = SparkSession\
        .builder\
        .appName("PythonNews")\
        .getOrCreate()

    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])

    articles = lines.map(lambda urls: getArticletText(urls))

    hashingTF = HashingTF()
    tf = hashingTF.transform(articles)

    tf.cache()
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf)
    
    rows = tfidf.zipWithIndex()
    
    bm = IndexedRowMatrix(rows.map(lambda row : IndexedRow(row[1], row[0]))).toBlockMatrix()

    #bm_t = bm.transpose()
    #result_mat = bm.multiply(bm_t)
    #exact = result_mat.toIndexedRowMatrix().toRowMatrix()

    exact = bm.transpose().toIndexedRowMatrix().columnSimilarities()

    print(exact.entries.collect())

    #print(exact.entries.collect()[0])

    #parsedArticles = articles.collect()

    #tfidf = TfidfVectorizer().fit_transform(parsedArticles)
    #pairwise_similarity = tfidf * tfidf.T
    #print(pairwise_similarity[0])

spark.stop()
