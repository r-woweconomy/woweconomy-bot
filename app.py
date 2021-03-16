from datetime import timedelta, date
import os
import praw
import re
import math
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import minmax_scale
from textdistance import jaro_winkler

for package in ['punkt', 'stopwords']:
    nltk.download(package, quiet=True)

# Credit to https://github.com/4OH4/doc-similarity/blob/master/tfidf.py
# for the search term to text relevance calculation

AUTHOR_WHITELIST = [ "AutoModerator" ]

reddit = praw.Reddit(client_id=os.environ.get('CLIENT_ID'),
                    client_secret=os.environ.get('CLIENT_SECRET'),
                    username=os.environ.get('USERNAME'),
                    password=os.environ.get('SECRET'),
                    user_agent='r/woweconomy bot')

class LemmaTokenizer:
    """
    Interface to the WordNet lemmatizer from nltk
    """
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]

def process_str(s):
    """
        Processes a string in the following manner:
            1. Removes all non-alphanumeric and non-whitespace characters
            2. Removes all stop words
    """
    stop_words = stopwords.words('english')
    clean = re.sub("[^\w\s]", "", s)
    # Must use this rather than set operation, because we need order preserved
    clean = [word for word in clean.split() if word not in stop_words]
    clean = " ".join(clean)
    return clean

def score_posts(search_terms, posts_corpus):
    # set up for text comparison
    # Credit: https://github.com/4OH4/doc-similarity/blob/master/tfidf.py
    tokenizer=LemmaTokenizer()
    stop_words = set(stopwords.words('english')) 
    stop_words = tokenizer(' '.join(stop_words))
    vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenizer)
    vectors = vectorizer.fit_transform([search_terms] + posts_corpus)
    cosine_similarities = linear_kernel(vectors[0:1], vectors).flatten()
    scores = [item.item() for item in cosine_similarities[1:]]  # convert back to native Python dtypes
    scores_normalized = minmax_scale(scores) if scores else scores

    return scores_normalized

def score_titles(submission_title, titles_corpus):
    titles_score = [jaro_winkler(submission_title, title) for title in titles_corpus]
    return titles_score

def find_relevant_posts(submission, subreddit, days=0, weeks=0, months=6, years=0):
    """
        Finds relevant posts for the given post title. Looks at posts from the
        given time period, compares titles and contents, and gives a weighted
        rank for relevancy
        Contents sorted in reverse order of relevance
        Returns a list of tuples of (Reddit.Submission, relevance_score)
    """

    # have to use OR keywords for reddit search
    subreddit_search = " OR ".join(process_str(submission.title).split())

    # filter or date later so we get a better corpus for text comparison
    authors_filter = AUTHOR_WHITELIST + [submission.author]
    posts = [post for post in reddit.subreddit('woweconomy').search(subreddit_search) if post.author not in authors_filter]

    scores_search = process_str(submission.title)

    posts_corpus = [post.selftext for post in posts]
    posts_scores = score_posts(process_str(submission.title), posts_corpus)

    titles_corpus = [process_str(post.title) for post in posts]
    titles_scores = score_titles(process_str(submission.title), titles_corpus)

    # weight title score more heavily than post score
    title_weight = .4
    post_weight = (1 - title_weight)
    assert title_weight <= 1 and post_weight >= 0
    weighted_scores = list(map(lambda ps,ts: (ps * post_weight) + (ts * title_weight),
        posts_scores, titles_scores))

    posts_scores = zip(posts, weighted_scores)

    # remove outdated posts
    date_limit = timedelta(days=days, weeks=weeks + (months * 4) + (years * 52))
    today = date.today()
    posts_scores = list(filter(lambda post_score: (date.fromtimestamp(post_score[0].created) > (today - date_limit)),
                        posts_scores))

    posts_scores.sort(key=lambda post_score: post_score[1], reverse=True)

    return posts_scores

def suggest_posts(submission, relevant_posts):

    rp_format_str = "- [{}]({}) ({:.0%})"
    suggestions_list = "\n".join([rp_format_str.format(post.title,post.url,score) for post,score in relevant_posts])
    message = f"""Hello, /u/{submission.author}!

Thank you for your post on /r/WoWEconomy. In an effort to cut down on duplicate posts, I've found some relevant submissions that might be of assistance:

{suggestions_list}

If, after viewing the above submissions, you've found the information you've been looking for, consider deleting your submission to help the subreddit reduce spam.

*If you're looking for resources, there's help for all manners of goblining in the [sidebar](https://www.reddit.com/r/woweconomy/about/sidebar) and on the [wiki](https://www.reddit.com/r/woweconomy/wiki).*
___
^[I'm&nbsp;a&nbsp;bot](https://github.com/r-woweconomy/woweconomy-bot)&nbsp;
"""

    submission.reply(message).mod.distinguish(sticky=True)
    submission.save()


def main():
    
    testwebot = reddit.subreddit('testwebot')
    woweconomy = reddit.subreddit('woweconomy')

    wenew = testwebot.new(limit=10)
    
    for submission in wenew:
        if not submission.link_flair_text or submission.saved:
            print(f"SAVED OR FLAIR: {submission.title}: {submission.url}")
            continue
        print(f"Finding relevant posts for: {submission.title}: {submission.url}")
        relevant_posts = find_relevant_posts(submission, woweconomy)
        if relevant_posts:
            above_threshold = list(filter(lambda ps: math.floor(ps[1] * 100) > 75, relevant_posts))
            if above_threshold:
                suggest_posts(submission, above_threshold)



if __name__ == '__main__':
    main()