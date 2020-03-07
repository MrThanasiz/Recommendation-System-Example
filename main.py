import time
import csv
import random
from operator import itemgetter, attrgetter
import os
from datetime import datetime
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

start_time = time.time()
ratedir = "BX-Book-Ratings.csv"
bookdir = "BX-Books.csv"
userdir = "BX-Users.csv"
minUserReviews = 5
minBookReviews = 10
topReviewsN = 3
usersN = 5
recommendationsN = 10
resultsPrecisionDigits = 2
weightJaccardKeywords = 0.2
weightJaccardAuthor = 0.4
weightJaccardYear = 0.4
weightDiceKeywords = 0.5
weightDiceAuthor = 0.3
weightDiceYear = 0.2



def loadFile(filedir): # Loads CSV files to memory
    out = []
    with open(filedir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                out.append(row)
                line_count += 1
                #if line_count <= 3: # prints the first 3 rows for debug
                #    print(row)
        #print(f'Processed {line_count} lines.')
        return out


def preprocessingA(users, books, reviews): # Removes users with less reviews than minUserReviews and books with less than minBookReviews
    usersDict = {}
    booksDict = {}
    # Dhmioyrgia 2 dictionairies me id xrhstwn kai vilviwn kai pluthos review
    for review in reviews:
        if review[0] in usersDict:
            usersDict[review[0]] = usersDict[review[0]] + 1
        else:
            usersDict[review[0]] = 1
        if review[1] in booksDict:
            booksDict[review[1]] = booksDict[review[1]] + 1
        else:
            booksDict[review[1]] = 1

    # Afairesh apo ta dictionairies osous user exoun katw apo 5 reviews kai osa vivlia katw apo 10

    tempDict = usersDict.copy()
    for user in usersDict:
        if usersDict[user] < minUserReviews:
            tempDict.pop(user)
    usersDict = tempDict.copy()

    tempDict = booksDict.copy()
    for book in booksDict:
        if booksDict[book] < minBookReviews or book[3] == 0:
            tempDict.pop(book)
    booksDict = tempDict.copy()

    # Telos afairoume osous den einai sto dict apo thn arxiki lista.
    usersOut = []
    for user in users:
        if user[0] in usersDict:
            usersOut.append(user)

    booksTemp = []
    for book in books:
        if book[0] in booksDict:
            booksTemp.append(book)

    # Epishs krataw mono tis prwtes 4 steiles (ISBN, Book-Title, Book-Author, Year-Of-Publication)
    # ws veltistopoihsh efoson mono autes xrhshmopoioume.
    booksOut = []
    for book in booksTemp:
        booksOut.append([book[0], book[1], book[2], book[3]])

    return usersOut, booksOut

def titleGenerateKeywords(bookTitle): # Keyword generation (Tokenize, Stop-word Removal)
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(bookTitle)
    stop_words = set(stopwords.words('english'))
    keywords = []
    for word in tokenized:
        if word not in stop_words: keywords.append(word)
    return keywords


def preprocessingB(books): # Generates list with book id and its keywords (based on the title)
    booksKeywords = {}
    for book in books:
        keywords = titleGenerateKeywords(book[1].lower())
        booksKeywords[book[0]] = keywords
    return booksKeywords



def getReviews(reviewCount, user, books, reviews): # Returns the top reviewCount number of reviews for the user, only if the books exist. With 0 returns all reviews
    #review [user,book,score]
    reviewList = []
    for review in reviews:
        if user[0] == review[0]:
            reviewList.append(review)
    reviewList.sort(key=itemgetter(2), reverse=True)
    reviewListOut = []
    for review in reviewList:
        for book in books:
            if review[1] == book[0]:
                reviewListOut.append(review)
    if reviewCount == 0: # if review count is 0 returns all reviews
        return reviewListOut
    if len(reviewListOut) >= reviewCount: # otherwise returns number specified or less
        return reviewListOut[:reviewCount]
    else:
        return reviewListOut

def reviewsEnough(user, books, review): # Gets reviews for user and returns if the reviews are sufficient to continue
    return len(getReviews(topReviewsN, user, books, review)) >= topReviewsN

def profileGeneration(user, books, reviews, bookKeywords): # Generates a profile for the user based on his top 3 books reviewed
    userProfile = [[],[],[]]
    userReviews = getReviews(topReviewsN, user, books, reviews)
    for review in userReviews:
        for book in books:
            if review[1] == book[0]:
                userProfile[0] = userProfile[0] + bookKeywords[book[0]]
                userProfile[1].append(book[2])
                userProfile[2].append(book[3])
    return userProfile

def yearScore(yearA, yearB):
    index = 1 - (abs(int(yearA) - int(yearB)) / 2005)
    return index

def diceCoefficient(listA, listB):
    setA = set(listA)
    setB = set(listB)
    setIntersection = set()
    for item in setA:
        if item in setB:
            setIntersection.add(item)
    index = 2 * len(setIntersection)/(len(setA) + len(setB))
    return index

def jaccardIndex(listA, listB):
    setA = set(listA)
    setB = set(listB)
    setIntersection = set()
    setUnion = set()
    for item in setA:
        setUnion.add(item)
    for item in setB:
        setUnion.add(item)
        if item in setA:
            setIntersection.add(item)
    index = len(setIntersection)/len(setUnion)
    return index

def getRecommendations(user, books, reviews, bookKeywords, algorithm): # Creates the recommendation list for the specified user with the specified algorithm ("Jaccard" / "Dice", Default: "Jaccard)
    userProfile = profileGeneration(user, books, reviews, bookKeywords)
    userAllReviews = getReviews(0, user, books, reviews)
    bookScoreTotal = []
    for book in books:
        bookScoreYear = yearScore(book[3], random.choice(userProfile[2]))
        if algorithm == "Dice":
            bookScoreKeywords = diceCoefficient(bookKeywords[book[0]], userProfile[0])
            bookScoreAuthor = diceCoefficient([book[2]], userProfile[1])
            scoreTotal = bookScoreKeywords * weightDiceKeywords + bookScoreAuthor * weightDiceAuthor + bookScoreYear * weightDiceYear
        elif algorithm == "Jaccard":
            bookScoreKeywords = jaccardIndex(bookKeywords[book[0]], userProfile[0])
            bookScoreAuthor = jaccardIndex([book[2]], userProfile[1])
            scoreTotal = bookScoreKeywords * weightJaccardKeywords + bookScoreAuthor * weightJaccardAuthor + bookScoreYear * weightJaccardYear
        else:
            print("IN:getRecommendations - Unknown algorithm, running Jaccard...")
            bookScoreKeywords = jaccardIndex(bookKeywords[book[0]], userProfile[0])
            bookScoreAuthor = jaccardIndex([book[2]], userProfile[1])
            scoreTotal = bookScoreKeywords * weightJaccardKeywords + bookScoreAuthor * weightJaccardAuthor + bookScoreYear * weightJaccardYear
        bookScoreTotal.append([book[0], scoreTotal])
    bookScoreTotal.sort(key=itemgetter(1), reverse=True)
    bookUnreviewedScore = bookScoreTotal.copy()
    for book in bookScoreTotal:
        for review in userAllReviews:
            if book[0] == review[1]:
                bookUnreviewedScore.remove(book)
    return bookUnreviewedScore[:recommendationsN]


def RecommendationSequence(users, books, reviews, bookKeywords): # Checks if reviews of user are enough and creates recommendations for the User (Jaccard & Dice)
    recommendationsJaccard = []
    recommendationsDice = []
    for x in range(usersN):
        user = random.choice(users)
        while not reviewsEnough(user, books, reviews):
            print("User or Book reviews not enough, Picking a new user...")
            user = random.choice(users)
        recommendationsJaccard.append(user[0])
        recommendationsDice.append(user[0])
        recommendationsJaccard.append(getRecommendations(user, books, reviews, bookKeywords, "Jaccard"))
        recommendationsDice.append(getRecommendations(user, books, reviews, bookKeywords, "Dice"))
    return recommendationsJaccard, recommendationsDice

def saveRecommendations(users, books, recommendations, algorithm): # Gets a recommendations list and the name of the algorithm and saves them to a file.
    if not os.path.exists("Results"):
        os.makedirs("Results")
    for i in range(0, len(recommendations), 2):
        file = open("Results/" + recommendations[i] + " " + algorithm + ".txt", "w+")
        for user in users:
            if str(user[0]) == recommendations[i]:
                file.write("User ID: " + user[0] + " Location: " + user[1] + " Age: " + user[2] + "\n")
        file.write("Top books: \n")
        for j in range(len(recommendations[i + 1])):
            for book in books:
                if book[0] == recommendations[i + 1][j][0]:
                    file.write("Book ID: " + book[0] + " Score: " + str(round(recommendations[i + 1][j][1], resultsPrecisionDigits)) + " Title: " +
                    book[1] + " Author: " + book[2] + " Year of Publication: " + book[3] + "\n")
        file.close()

def rankBiasedOverlapSingle(listA, listB):
    count = min(len(listA),len(listB))
    total = 0
    intersect = 0
    for i in range(0,count):
        for j in range(0,i+1):
            if listA[i][0] == listB[j][0]:
                intersect = intersect + 1
        total = total + intersect / (i+1)
    result = total / count
    return result

def compareOverlapUsersN(recommendationListA,recommendationListB, algorithmA, algorithmB): # Calculates rank biased overlap between two recommendation lists and saves it to a file.
    if not os.path.exists("Results"):
        os.makedirs("Results")
    timestamp = datetime.now()
    currentTime = str(timestamp)[:19].replace(":",".")
    file = open("Results/" + currentTime + "-Run Overlap " + algorithmA + " - " + algorithmB + ".txt", "w+")
    minListLen = min(len(recommendationListA),len(recommendationListB))
    for i in range(0, minListLen, 2):
        file.write("UserID: " + str(recommendationListA[i]) + " Overlap: " + str(round(rankBiasedOverlapSingle(recommendationListA[i+1],recommendationListB[i+1]), resultsPrecisionDigits)) + "\n")
    file.close()

def createGoldenStandardRecommendationList(recommendationListA, recommendationListB): # Creates Golden Standard list, based on two other lists.
    minList = min(len(recommendationListA),len(recommendationListB))
    tempListDouble = []
    tempListSingle = []
    goldenStandardList = []
    for i in range(0, minList, 2):
        for itemA in recommendationListA[i+1]:
            for itemB in recommendationListB[i+1]:
                if itemA[0] == itemB[0]:
                    itemC = [itemA[0], itemA[1] + itemB[1]]
                    tempListDouble.append(itemC)
        for itemA in recommendationListA[i+1]:
            exists = False
            for itemC in tempListDouble:
                if itemA[0] == itemC[0]:
                    exists = True
            if not exists:
                tempListSingle.append(itemA)
        for itemB in recommendationListB[i+1]:
            exists = False
            for itemC in tempListDouble:
                if itemB[0] == itemC[0]:
                    exists = True
            if not exists:
                tempListSingle.append(itemB)

        tempListDouble.sort(key=itemgetter(1), reverse=True)
        tempListSingle.sort(key=itemgetter(1), reverse=True)
        tempListFinal = tempListDouble + tempListSingle
        goldenStandardList.append(recommendationListA[i])
        goldenStandardList.append(tempListFinal)
        tempListUsers = []
        tempListDouble = []
        tempListSingle = []
    return goldenStandardList


nltk.download('stopwords') # meta thn prwth ektelesh den xreiazetai, katevazei thn lista me ta stopwords
print("Data Loading Started...")
users   = loadFile(userdir)
books   = loadFile(bookdir)
reviews = loadFile(ratedir)
print("Data Loading Complete, Preprocessing Started...")
usersPrepped, booksPrepped = preprocessingA(users, books, reviews)
bookKeywords = preprocessingB(booksPrepped)
print("Preprocessing Complete, Recommendations Generation Started...")
recommendationsJaccard, recommendationsDice = RecommendationSequence(usersPrepped, booksPrepped, reviews, bookKeywords)
print("Recommendations Generation Complete, Saving Results...")
saveRecommendations(usersPrepped, booksPrepped, recommendationsJaccard, "Jaccard")
saveRecommendations(usersPrepped, booksPrepped, recommendationsDice, "Dice")
print("Saving Results Complete, Calulating and Saving Overlap...")
compareOverlapUsersN(recommendationsJaccard,recommendationsDice, "Jaccard",  "Dice")
print("Overlap Saving Complete, Calulating 'Golden Standard' List...")
recommendationsGoldenStandard = createGoldenStandardRecommendationList(recommendationsJaccard,recommendationsDice)
print("Calulating 'Golden Standard' List Complete, Calculating and Saving 'Golden Standard' List Overlap...")
compareOverlapUsersN(recommendationsGoldenStandard, recommendationsJaccard, "Golden Standard",  "Jaccard")
compareOverlapUsersN(recommendationsGoldenStandard, recommendationsDice, "Golden Standard",  "Dice")
saveRecommendations(usersPrepped, booksPrepped, recommendationsGoldenStandard, "GS")
print("Saving 'Golden Standard' List Overlap Complete, Exiting...")
print("Total Execution Time: " + str(round(time.time() - start_time, resultsPrecisionDigits)))
