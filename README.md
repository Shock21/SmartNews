# SmartNews

Running the serial version is done through the simple command: python3 source
Prereq: Python3, Newspaper Python3 Lib 

Running the distributed version is done through the following command: ./bin/spark-submit --master local[2] source input
Where the input is a file that contains links to the articles and the firest article is the one used as reference.
Prereq: Apache Spark(PySpark), Python3, Newspaper Python3 Lib 
