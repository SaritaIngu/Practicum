print("NLP Based News Articler Recommender")

#import libraries
import seaborn as sns
import matplotlib as plt
import bs4 as bs
import csv
import requests
import urllib
from urllib.request import urlopen
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from datetime import date, timedelta
from datetime import datetime
from dateutil import parser
from IPython.display import HTML
import pytwits
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as sentintentanalyzer

pd.set_option('display.max_colwidth',200)
num_of_days = 3
last_date = date.today() - timedelta(days=(num_of_days-1))

def GetWSJContent(article_link):
    import bs4 as bs
    import csv
    import requests
    import urllib
    import re
    
    strContent=''
    
    from urllib.request import urlopen
    url = article_link
     #print(url)
 
    #req = urllib.request.Request(url, headers={'User-Agent': 'IE9'})
    with urllib.request.urlopen(url) as f:
        #sourcehtml = urlopen(req)
        #print(sourcehtml)
        soup = bs.BeautifulSoup(f.read().decode('utf-8'),"lxml")        
        #print(soup)
        
        if url.find("marketwatch") != -1:
            BulletsDiv = soup.find("div", {"id": "article-body"})
            if BulletsDiv is not None:
               pbullets = BulletsDiv.findAll("p")
               for bullet in pbullets:
                  bulletHolder = bullet.get_text().strip() 
                  strContent = strContent + bulletHolder  
                  strContent = strContent.replace("\r","")
                  strContent = strContent.replace("\n","")
                  strContent = strContent.replace('  ', ' ')
                  re.sub(' +', ' ',strContent)
        elif url.find("barrons") != -1 :
            BulletsDiv = soup.find("div", {"itemprop": "articleBody"})
            if BulletsDiv is not None:
               pbullets = BulletsDiv.findAll("p")
               for bullet in pbullets:
                  bulletHolder = bullet.get_text().strip() 
                  strContent = strContent + bulletHolder  
                  strContent = strContent.replace("\r","")
                  strContent = strContent.replace("\n","")
                  strContent = strContent.replace('  ', ' ')
                  re.sub(' +', ' ',strContent)
                
    return strContent  

def GetReutersContent(article_link):
    strContent=''    
    url = article_link
    req = urllib.request.Request(url, headers={'User-Agent': 'IE9'})
    sourcehtml = urlopen(req)   
    soup = bs.BeautifulSoup(sourcehtml,"lxml")
    #print(soup)
    ReutersMaindiv = soup.find("div", {"class":"StandardArticleBody_body"})  
    #print(ReutersMaindiv)
    pbullets = ReutersMaindiv.findAll("p")    
    for bullet in pbullets:
        bulletHolder = bullet.get_text().strip() 
        strContent = strContent + bulletHolder     
    
    return strContent  

def GetYahoofinContent(article_link):
    strContent=''    
    url = article_link
    req = urllib.request.Request(url, headers={'User-Agent': 'IE9'})
    sourcehtml = urlopen(req)   
    soup = bs.BeautifulSoup(sourcehtml,"lxml")
    pbullets = soup.findAll("p")    
    for bullet in pbullets:
        bulletHolder = bullet.get_text().strip() 
        strContent = strContent + bulletHolder 
                
    return strContent 

def ConvertDate(dt = "Today"):
   """Return date in mm/dd/yyyy format"""
   
   if dt.find("Today") != -1:
        #try:
          dts = dt.split(" ")
          hm = dts[1]
          m = hm.split(':')[1]
          h = hm.split(':')[0]
          ampm = dts[2]   
          #print(h)
          if ampm =="PM":                         
             h = int(h) 
             if h < 12:
                h = h+12
          today2 = datetime(date.today().year, date.today().month, date.today().day, int(h), int(m), 0)
          #print(today2)          
          return today2.strftime('%m/%d/%Y  %H:%M')
        #except: return ''
   elif dt.find("Yesterday") != -1:
        try:
          yest = date.today() - timedelta(days=1)
          dts = dt.split(" ")
          hm = dts[1]
          m = hm.split(':')[1]
          h = hm.split(':')[0]
          ampm = dts[2]        
          if ampm =="PM":             
             h = int(h) 
             if h < 12:
                h = h+12
          yest2 = datetime(yest.year, yest.month, yest.day, int(h), int(m), 0)          
          return yest2.strftime('%m/%d/%Y %H:%M')
        except: return ''
   elif dt.find(",") != -1 :
       try:
         dts = dt.split(",")
         if(len(dts)>=2):
           dtnew =  dts[1].strip()+ ' ' + str(datetime.now().year)
           #print(dtnew)
         dt_new_format = datetime.strptime(dtnew, '%b. %d %Y')        
         return dt_new_format.strftime("%m/%d/%Y")
       except: return ''
   elif ((dt.find("hours") != -1) | (dt.find("hour") != -1)) :
        try:
          dts = dt.split(" ")
          #print(dts[0])
          today = datetime.now() - timedelta(hours=int(dts[0]))         
          return today.strftime('%m/%d/%Y %H:%M')
        except: return ''
   elif dt.find("min") != -1 :
       try:
        dts = dt.split(" ")
        today = datetime.now() - timedelta(minutes=int(dts[0]))        
        return today.strftime('%m/%d/%Y %H:%M')
       except: return ''
   elif dt.find("am") != -1 :
       try:
        dts = dt.split("am")
        dts1 = dts[0].split(":")
        today = datetime.today()      
        todaydt = datetime(today.year, today.month, today.day, int(dts1[0]), int(dts1[1]), 0) 
        return todaydt.strftime('%m/%d/%Y %H:%M')
       except: return ''
   elif dt.find(str(datetime.now().year)) != -1 : 
       try:
          dt1 = datetime.strptime(dt, '%b %d %Y')
          return dt1.strftime('%m/%d/%Y')
       except: return ''
   elif dt.find('/') != -1 :
       try:
         return dt
       except:           
           return datetime.today()
   else: 
        return datetime.today()

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')

def Sentiment(a):
    import pysentiment as ps
    lm = ps.LM()
    tokens = lm.tokenize(a)
    a= lm.get_score(tokens)['Polarity']
    if a>0:
        return "Positive" 
    elif(a==0):
        return "Neutral" 
    else:
        return "Negative" 
    
def Score(a):
    import pysentiment as ps
    lm = ps.LM()
    tokens = lm.tokenize(a)
    a= lm.get_score(tokens)['Polarity']
    return a

def t_sentiment(tweet):
    analyze = SentimentIntensityAnalyzer()
    comp = analyze.polarity_scores(tweet)['compound']
    if comp > 0:
        return "Positive"
    elif comp < 0:
        return "Negative"
    else:
        return "Neutral"

def t_polarity(tweet):
    analyze = SentimentIntensityAnalyzer()
    return analyze.polarity_scores(tweet)['compound']


#------------ Wall Street Journal News Parser-------------
#AllArticles = ['Ticker','Date','Title','Link','Text']
AllArticles = []
cols=['Ticker','Date','Title','Link','Text']
df_articles = pd.DataFrame([],columns=cols)
#stocks = ["TSLA","AAPL","AMZN", "GOOG", "FB"]
stocks = ["TSLA","AAPL","AMZN"]
#stocks = ["TSLA"]
print("-----------Scraping WSJ Articles------------")
for stock in stocks :
    url = "https://quotes.wsj.com/"+stock
    req = urllib.request.Request(url, headers={'User-Agent' : "foobar"})
    sourcehtml = urlopen(req)
    soup = bs.BeautifulSoup(sourcehtml.read().decode('utf-8'),"lxml")  

#Parse the document using soup object and extract the required text 
    mydivs = soup.findAll("span", {"class":"headline"}) 
    mydiv_time = soup.findAll("li", {"class":"cr_dateStamp"})
    
    mydiv_url = soup.findAll("span", {"class":"headline"})
    
    print("-----------"+stock+"------------")

    for i,div in enumerate(mydivs):       
        atextHolder = div.findAll('a')
        articleHolder = atextHolder[0].get_text().strip()
        article_url = mydiv_url[i].find('a').attrs['href']
        article_time = mydiv_time[i].get_text()
        article_FormattedDate  = ConvertDate(article_time)   

        try:
           article_date = parser.parse(article_FormattedDate)
        except:           
           article_date = datetime.now()

        if  article_date.date() < last_date :
            break;
        print(article_time+" -  "+articleHolder)
        print(article_url)
        article_content= GetWSJContent(article_url)
        lstarticle = [stock, article_FormattedDate, articleHolder, article_url, article_content] 
        df2 = pd.DataFrame([lstarticle], columns=cols)        
        df_articles = df_articles.append(df2)
        AllArticles.append(lstarticle)       
 
#------------------Reuters Parser------------------
print("-----------Scraping Reuters Articles------------")
for stock in stocks :   
    url = "https://www.reuters.com/finance/stocks/overview/"+stock +".OQ"
    req = urllib.request.Request(url, headers={'User-Agent' : "foobar"})
    sourcehtml = urlopen(req)
    soup = bs.BeautifulSoup(sourcehtml,"lxml")
    
#Parse the document using soup object and extract the required text 
    Newsdiv = soup.find("div", {"id":"companyOverviewNews"})   
    ModuleDiv = Newsdiv.find("div", {"class":"moduleBody"}) 
    FeatureDivs = ModuleDiv.findAll("div", {"class":"feature"}) 
    #print(len(FeatureDivs))
    
    
    mydiv_url = soup.findAll("span", {"class":"headline"})
    
    print("-----------"+stock+"------------")

    for i,div in enumerate(FeatureDivs): 
        atextHolder = div.findAll('a')       
        articleHolder = atextHolder[0].get_text().strip()        
        article_url =  "https://www.reuters.com/"+ atextHolder[0].attrs['href']   
        div_time = div.find("div", {"class":"relatedInfo"})
        span_time = div_time.find("span", {"class":"timestamp"}) 
        article_time = span_time.get_text().strip()
        article_FormattedDate = ConvertDate(article_time)

        try:
           article_date = parser.parse(article_FormattedDate)
           print(article_date)
        except:           
           article_date = datetime.now()
           print('in except')

        if  article_date.date() < last_date :
            break;
        print(article_time+" -  "+articleHolder)
        print(article_url)
        article_content = GetReutersContent(article_url)
        lstarticle = [stock, article_FormattedDate, articleHolder, article_url, article_content]       
        df2 = pd.DataFrame([lstarticle], columns=cols)        
        df_articles = df_articles.append(df2)
        AllArticles.append(lstarticle)
        

#------------------Yahoo Finance------------------
for stock in stocks :
    url = "https://finance.yahoo.com/quote/"+stock+"/news?p="+stock
    req = urllib.request.Request(url, headers={'User-Agent' : "foobar"})
    sourcehtml = urlopen(req)
    soup = bs.BeautifulSoup(sourcehtml,"lxml")

#Parse the document using soup object and extract the required text 
    mydivs = soup.findAll("h3", {"class":"Mb(5px)"}) 
    mydiv_time = soup.findAll("div", {"class":"C(#959595) Fz(11px) D(ib) Mb(6px)"})
    mydiv_url = soup.findAll("span", {"class":"headline"})   
    print("----------------------------------"+stock+"------------------------------------")
    
    for i,div in enumerate(mydivs):  
        atextHolder = div.findAll('a')
        articleHolder = atextHolder[0].get_text().strip()
        articlehref = atextHolder[0]['href']
        articleLink = articlehref
        artTime = mydiv_time[i].get_text().split('•')[1:2]
        strtime = ''.join(artTime)
        article_time =  datetime.now()
        
        #format article time
        today = datetime.now()
        if strtime.find("minute")!= -1:
            minute = int(strtime[:2])
            article_time = datetime.now() - timedelta(minutes=minute)
            acttime = (format(datetime.now() - timedelta(minutes=minute), "%m-%d-%Y %I:%M%p"))
            print (acttime)
        elif strtime.find("hour")!= -1:
            hour = int(strtime[:2])
            article_time = datetime.now() - timedelta(hours=hour)
            acttime = (format(datetime.now() - timedelta(hours=hour), "%m-%d-%Y %I:%M%p"))
            print (acttime)
        elif strtime.find("yesterday")!= -1:
            article_time = datetime.now() - timedelta(days=1)
            acttime =(format(datetime.now() - timedelta(days=1),"%m-%d-%Y %I:%M%p"))
            print (acttime)
        else:
            acttime = (format(strtime, "%m-%d-%Y %I:%M%p"))
            print (acttime)
           
        print (articleHolder)
       
        if  article_time.date() < last_date :
            break;
        
        #url = mydiv_url[i].find('a').attrs['href']
        if articlehref.find(".com") != -1:
            print(articlehref)
        else:
            articleLink = 'https://finance.yahoo.com'+articlehref
            print('https://finance.yahoo.com'+articlehref)
        print("\n")
        article_content= GetYahoofinContent(articleLink)
        #print(article_content)
        lstarticle = [stock, acttime, articleHolder, articleLink, article_content]
        df2 = pd.DataFrame([lstarticle], columns=cols)        
        df_articles = df_articles.append(df2)
        AllArticles.append(lstarticle)    


filename = 'AllArticlesOutput'+ datetime.now().strftime('%Y-%m-%d-%H-%M') +'.csv'
with open(filename,'w', newline='') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerow(['Ticker','Date','Title','Link','Text'])
    # Write Data to File
    for item in AllArticles:
       try:
          wr.writerow(item)
       except:           
          continue

# Removing Duplicates (Similarity scoring)
similarity_threshold = 0.5
vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english', use_idf=False, ngram_range=(1,2))

for stock in stocks:
    df1 = df_articles[df_articles.Ticker == stock]
    #dfname = "df_"+stock+"_articles"
    df_stock_articles = df1[df1['Text']!='']
    print('Articles With Content: ' +str(len(df_stock_articles)) + " for "+stock)
    print("-----------Removing Duplicates------------")
    articles = df_stock_articles['Text'].sort_index(ascending = False)
    dtm = vectorizer.fit_transform(articles)
    doc_term_matrix = pd.DataFrame(dtm.toarray(),index=articles,columns=vectorizer.get_feature_names())
    similarity = np.asarray(np.asmatrix(doc_term_matrix) * np.asmatrix(doc_term_matrix).T)
    similarity = np.triu(similarity, k=1)
    df_sim = pd.DataFrame(similarity, columns = articles)
    unique_articles = df_sim[(df_sim<=similarity_threshold)].dropna(axis = 1, how = 'any').columns.values
    df_articles['Unique_'+stock] = df_articles['Text'].apply(lambda x: 1 if x in unique_articles else 0)

col_list = [col for col in df_articles.columns if 'Unique' in col]
df_articles['Unique'] = df_articles[col_list].sum(axis=1)
df_unique = df_articles[df_articles['Unique']>=1]
df_unique= df_unique[df_unique.columns.drop(list(df_unique.filter(regex='Unique')))]
print('Number Of Unique Articles: ' +str(len(df_unique)))

#sentiment scoring
print("-----------Sentiment Scoring------------")
df_unique['Sentiment']=df_unique['Text'].apply(lambda x: Sentiment(x))
df_unique['Score']=df_unique['Text'].apply(lambda x: Score(x)) 

print("-----------Creating html file------------")
#write html file
df_unique.to_html('ImportantArticles'+ datetime.now().strftime('%Y-%m-%d-%H-%M') +'.html')


## Scraping tweets from StockTwits with API authentication
tweets = []
tickers = []
access_token = '26c8e5b3d7bd2903f6091364ae85a91cce9b9aa4'
stocktwits = pytwits.StockTwits(access_token=access_token) # This would also work without passing token.

for stock in stocks:
    symbol, cursor, messages = stocktwits.streams(path = 'symbol', id = stock) # set limit on messages by limit = __, max = 30)
    for msg in messages:
        tweets.append(msg.body)
        tickers.append(symbol.symbol)    
tweet_df = pd.DataFrame({'Ticker': tickers, 'Tweet': tweets})

#Sentiment scoring of tweets
tweet_df['Tweet'] = tweet_df['Tweet'].str.replace("&#39;", "'")
tweet_df['Tweet'] = tweet_df['Tweet'].str.replace("$", "")
tweet_df['Tweet'] = tweet_df['Tweet'].str.replace('&quot;', '"')
tweet_df['Tweet'] = tweet_df['Tweet'].str.replace("&amp;", "&")
tweet_df['Sentiment'] = tweet_df['Tweet'].apply(lambda x: t_sentiment(x))
tweet_df['Score'] = tweet_df['Tweet'].apply(lambda x: t_polarity(x))
tweet_summary = tweet_df.groupby(['Ticker', 'Sentiment'], as_index=False).agg({'Tweet':'count','Score':'mean'}).rename(columns = {'Tweet':'Tweet_count', 'Score':'Avg_Score'})


##filtration
final_output_str = ''
pd.set_option('display.max_colwidth', 200)
for stock in stocks:
    positive = str(len(df_unique[(df_unique['Sentiment'] == 'Positive') & (df_unique['Ticker'] == stock)]))
    negative = str(len(df_unique[(df_unique['Sentiment'] == 'Negative') & (df_unique['Ticker'] == stock)]))
    neutral = str(len(df_unique[(df_unique['Sentiment'] == 'Neutral') & (df_unique['Ticker'] == stock)]))
    
	pos_tweets = str(tweet_summary[(tweet_summary['Ticker'] == stock) & (tweet_summary['Sentiment'] == 'Positive')].Tweet_count)
    neg_tweets = str(tweet_summary[(tweet_summary['Ticker'] == stock) & (tweet_summary['Sentiment'] == 'Negative')].Tweet_count)
    neu_tweets = str(tweet_summary[(tweet_summary['Ticker'] == stock) & (tweet_summary['Sentiment'] == 'Neutral')].Tweet_count)	
	
    df_stock = df_unique[df_unique['Ticker'] == stock]
    df_stock['abs_sentiment'] = df_stock['Score'].abs()
    df_stock =df_stock.sort_values(by= 'abs_sentiment', ascending=False).head(5)
    df_final= df_stock.drop('abs_sentiment', axis=1)
    print("-----------Creating html file------------")
    print("for" + stock)
    #write html file
    html_file_nm = 'ImportantArticles'+ datetime.now().strftime('%Y-%m-%d-%H-%M') +'.html'
    df_final.to_html(html_file_nm)
    ### to_html(feed name of file)

    ###### CREATE HTML CONTENT ###########
    with open(html_file_nm, 'r') as f:
        html_content_full = f.readlines()
        ####readlines: reads the entire file and converts it to a list
    print(html_content_full)

    new_html_content_full = []

    for elem in html_content_full:
      ##find to search
      if elem.find('table') > 0:
        ##if it finds table it will manipulate elem to add css
        re = '<table border="1" style="color:black;background-color:lavender" class="dataframe">\n'
        new_html_content_full.append(re)
      elif elem.find('http:') > 0:
        index_start = elem.index('>')
        end_index = elem.index('</td>')
        ##one quote for href and one for python string
        re = '<td>' + '<a  href="' + elem[index_start+1:end_index] + '">' +  elem[index_start+1:end_index] + '</a>'
        new_html_content_full.append(re)
      elif elem.find('https:') > 0:
        index_start = elem.index('>')
        end_index = elem.index('</td>')
        ##one quote for href and one for python string
        re = '<td>' + '<a  href="' + elem[index_start+1:end_index] + '">' +  elem[index_start+1:end_index] + '</a>'
        new_html_content_full.append(re)
      else:
        new_html_content_full.append(elem)

    html_content = ''
    html_content += '<body style="background-image: url(http://wall2born.com/data/out/365/image-48994060-beige-and-gray-wallpaper.jpg)">'+'</body><br>'
    html_content += '<h1 style="background-color:lightsteelblue; border:solid lavender; opacity:0.7;text-align:center, border:double">' + stock + '</h1></br>'
    html_content += \
    '<div style="background-color:lavender; border:double;float: left; width:10%; text-align:center;height:4%; padding-top:0.5%">Positive:' + positive + '</div><div style="float: left;"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;</div>' \
    '<div style="background-color:lavender; border:double;float: left; width:10%; text-align:center;height:4%; padding-top:0.5%"">  Negative:' + negative + '</div><div style="float: left;"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;</div>' \
    '<div style="background-color:lavender; border:double;float: left; width:15%; text-align:center;height:4%; padding-top:0.5%""> Neutral:' + neutral + '</div><br/><br/><br/><br/><br/><br/>'
    new_html_content_full.insert(0,html_content)
    final_output_str += ''.join(new_html_content_full)
    final_output_str += '<br/><br/><br/><br/>'

with open('ImportantArticles'+ datetime.now().strftime('%Y-%m-%d-%H-%M') +'.html', 'w+') as f:
        f.write(final_output_str)



