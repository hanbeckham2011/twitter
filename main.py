import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, StopWordsRemover
from pyspark.ml.classification import LogisticRegression

if __name__ == '__main__':
    spark = SparkSession.builder.config("spark.driver.memory", "6g").appName('data_processing').getOrCreate()

    df_train = spark.read.csv(r'./training.1600000.processed.noemoticon.csv', inferSchema=True, header=False)
    df_test = spark.read.csv(r'./testdata.manual.2009.06.14.csv', inferSchema=True, header=False)
    df_tweets = spark.read.csv(r'./pulled_tweets.csv', inferSchema=True, header=False)

    newColumns = ["polarity", "id", "date", "query", "user", "tweet"]
    df_train = df_train.toDF(*newColumns).dropna().where("polarity==0 or polarity ==4")
    df_train.drop("id", "date", "query", "user")
    df_test = df_test.toDF(*newColumns).dropna().where("polarity==0 or polarity ==4")
    df_test.drop("id", "date", "query", "user")
    newColumns = ['polarity', 'tweet']
    df_tweets = df_tweets.toDF(*newColumns)

    #df_train.show(truncate=False, n=5)
    #df_test.show(truncate=False, n=5)

    #print('tokenizer')
    tokenizer = Tokenizer(inputCol="tweet", outputCol="words") # separate tweets into individual words
    tokenizedTrain = tokenizer.transform(df_train)
    #tokenizedTrain.show(truncate=False, n=5)

    #print('stop words')
    swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="MeaningfulWords")
    SwRemovedTrain = swr.transform(tokenizedTrain)
    #SwRemovedTrain.show(truncate=False, n=5)

    #print('computing features...')
    hashTF = HashingTF(inputCol=swr.getOutputCol(), outputCol="features")
    numericTrainData = hashTF.transform(SwRemovedTrain).select('polarity', 'MeaningfulWords', 'features')
    #numericTrainData.show(truncate=False, n=5)

    #print('training...')
    lr = LogisticRegression(labelCol="polarity", featuresCol="features", maxIter=5, regParam=0.01)
    model = lr.fit(numericTrainData)
    #print("Training is done!")

    #print('tokenizing...')
    tokenizedTest = tokenizer.transform(df_test)
    #print('removing stop words...')
    SwRemovedTest = swr.transform(tokenizedTest)
    #print('computing features...')
    numericTest = hashTF.transform(SwRemovedTest).select('polarity', 'MeaningfulWords', 'features')
    #numericTest.show(truncate=False, n=5)

    print('testing model on test set...')
    prediction = model.transform(numericTest)
    predictionFinal = prediction.select("MeaningfulWords", "prediction", "polarity")
    #predictionFinal.show(n=5, truncate=False)
    correctPrediction = predictionFinal.filter(predictionFinal['prediction'] == predictionFinal['polarity']).count()
    totalData = predictionFinal.count()
    print("correct prediction:", correctPrediction, ", total data:", totalData,", accuracy:", correctPrediction / totalData)

    print('testing model on pulled tweets')
    #print('tokenizing...')
    tokenizedLiveTest = tokenizer.transform(df_tweets)
    #tokenizedLiveTest.show(n=10, truncate=False)
    #print('removing stop words...')
    SwRemovedLiveTest = swr.transform(tokenizedLiveTest)
    #SwRemovedLiveTest.show(n=10, truncate=False)
    #print('computing features...')
    numericLiveTest = hashTF.transform(SwRemovedLiveTest).select('polarity', 'MeaningfulWords', 'features')
    #numericLiveTest.show(n=10, truncate=False)
    prediction = model.transform(numericLiveTest)
    #predictionFinal = prediction.select("MeaningfulWords", "prediction")
    predictionFinal = prediction.select("prediction").toDF('prediction')

    final_output = pd.concat([df_tweets[['tweet']].toPandas(), predictionFinal[['prediction']].toPandas()], axis=1, ignore_index=True)
    print(final_output)