
# Recommendation-System-Example
Part of an assignment for Information Retrieval and Search Engines Module University of Macedonia

## Description

Simple example of a Content-Based Recommendation System for books, written in Python.

Data used can be found  here: [http://www2.informatik.uni-freiburg.de/~cziegler/BX/](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

Code preprocesses the data by removing books with less than 10 ratings, and users with less than 5.

Then generates a book profile based on the title of the book, and a user profile based on the title, year and author of the users 3 top rated books, then recommends 10 books to the user based on resemblance with the book profiles.

On each execution the program picks 5 random users and generates recommendations.
All of these numbers can be changed.

## Prerequisites

- Python
- The csv input files, in the same folder with the code ([can be found here](http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip).)
- [NLTK Package](https://www.nltk.org/install.html)  installed in Python.
- Internet connection for the first execution so it downloads the stopwords list used.
