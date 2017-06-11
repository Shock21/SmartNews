import newspaper
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer

#sourceList = ['http://cnn.com', 'http://bbc.com', 'http://nbc.com']
cnn_paper = newspaper.build('http://cnn.com', memoize_articles=False)


#for article in cnn_paper.articles:
#	if "trump" in article.url:
#		article.download()
#		article.parse()
#		print(article.text)
#		break

articles = [Article(url='http://www.nbcnews.com/politics/donald-trump/president-trump-unleashes-fired-fbi-director-james-comey-fake-media-n758341', language='en', fetch_images=False), 
Article(url='http://www.bbc.com/news/world-us-canada-39899542', language='en', fetch_images=False), Article(url='http://edition.cnn.com/2017/05/12/politics/donald-trump-james-comey-threat/index.html', language='en', fetch_images=False),
Article(url='http://edition.cnn.com/2017/05/12/health/ryan-dant-college-graduation-mps-rare-disease-profile/index.html', language='en', fetch_images=False)]

documentList = []
for article in articles:
	article.download()
	article.parse()
	documentList.append(article.text)


tfidf = TfidfVectorizer().fit_transform(documentList)
pairwise_similarity = tfidf * tfidf.T
print(pairwise_similarity)
	

