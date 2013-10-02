package pegasos

import math._
import scala.collection.mutable.Map
import scala.collection.mutable.HashMap

object Main {
	def main(args: Array[String]) {
		val start = System.currentTimeMillis
		val trainingSize = 4000
		val N = List(100,200,400,800,2000,4000)
		val data = getDataFromFile("data/spam_train.txt")
		val emailList = Support.getEmailList(data)
		val tmp = emailList.splitAt(trainingSize) /*splits training data set for validation set*/
		val trainingDataSet = tmp._1
		val validationDataSet = tmp._2
		val vocabTally = Support.buildVocabulary(trainingDataSet)
		var vocabList = new Array[String](vocabTally.size)
		var vocabIndex = 0
		for(word <- vocabTally) { /*converts to simple array for faster iteration*/
			vocabList(vocabIndex) = word._1
			vocabIndex+=1
		}
		val trainingFeatureVectors = Support.makeFeatureVector(trainingDataSet,vocabList)
		printPerceptronData(false,true,trainingFeatureVectors,validationDataSet,vocabList,1000)

		val algTime = System.currentTimeMillis

		println("\nTotal Learning Time " + (algTime-start)/1000.0 + " seconds")
		
		for(n <- N) { /* prints out results for N=100,200,400,800,2000,4000 */
			val nFeatureVectors = trainingFeatureVectors.splitAt(n)._1
			printPerceptronData(false,false,nFeatureVectors,validationDataSet,vocabList,1000)
			printPerceptronData(true,false,nFeatureVectors,validationDataSet,vocabList,1000)
		}
		
		println("===============================\n\nRUN WITH TEST DATA AND FULL TRAINING SET")
		val newVocabTally = Support.buildVocabulary(emailList)
		var newVocabList = new Array[String](newVocabTally.size)
		var newVocabIndex = 0
		for(word <- newVocabTally) { /*converts to simple array for faster iteration*/
			newVocabList(newVocabIndex) = word._1
			newVocabIndex+=1
		}
		val finalFeatureVectors = Support.makeFeatureVector(emailList,newVocabList)
		val testData = getDataFromFile("data/spam_test.txt")
		val testDataSet = Support.getEmailList(testData)
		printPerceptronData(true,false,finalFeatureVectors,testDataSet,newVocabList,15)

		
		val end = System.currentTimeMillis

		println("Total Running Time of all Tests: " + (end-start)/1000.0 + " seconds")
	}

	def printPerceptronData(averaged: Boolean,printWeights: Boolean,featureVectors: List[(Array[Int],Int)],
		validationDataSet: List[(Int,String)], vocabList: Array[String],iterMax: Int) = {

		val trainingData = if (averaged) { Perceptron.averagePerceptronTrain(featureVectors,iterMax) }
		else { Perceptron.perceptronTrain(featureVectors,iterMax) }
		val weights = trainingData._1
		val k = trainingData._2
		val iter = trainingData._3
		val algType = if(averaged) "Average Perceptron Train" else "Regular Perceptron Train"
		println("\n" + algType + " of with size N of " + featureVectors.size + "\n")
		println("Total Training Errors:" + k + ", Iterations:" + iter)
		val validationTrainingVectors = Support.makeFeatureVector(validationDataSet,vocabList)
		val testError = Perceptron.perceptronTest(validationTrainingVectors,weights)
		println(f"Test Error: $testError%1.3f")
		if (printWeights){
			printHeaviestWeights(weights,vocabList)
		}
	}

	def getDataFromFile(file: String) : String = {
		import scala.io.Source 
		val source = Source.fromFile(file)
		val lines = source.getLines mkString ""
		return lines.replaceAll("number"," number ")
	}

	def printHeaviestWeights(weights:Array[Double],vocabList: Array[String]) = {
		val heavyWeights = getHeaviestWeights(weights,vocabList)
		val mostPositiveWeights = heavyWeights._1
		val mostNegativeWeights = heavyWeights._2
		println("\nMost Positive Weights:\n")
		mostPositiveWeights.foreach {x => println(x._1 + ":" + x._2)}
		println("\nMost Negative Weights:\n")
		mostNegativeWeights.foreach {x => println(x._1 + ":" + x._2)}
	}

	def getHeaviestWeights(weights: Array[Double],vocabList: Array[String]) : (List[(String,Double)],List[(String,Double)]) = {
		var size = 15
		var mostPositveWeights = new Array[Double](size)
		var mostNegativeWeights = new Array[Double](size)
		var mostPositveWeightsIndex = new Array[Int](size)
		var mostNegativeWeightsIndex = new Array[Int](size)
		var mostPositive = List[(String,Double)]()
		var mostNegative = List[(String,Double)]()

		for (i <- 0 until mostNegativeWeights.length) {mostNegativeWeights(i) = 10}

		for(x <- 0 until weights.length) {
			var i = 0
			var j = 0
			while( i < mostPositveWeights.length && weights(x) > mostPositveWeights(i)) {
				if (i > 0) {
					mostPositveWeights(i-1) = mostPositveWeights(i)
					mostPositveWeightsIndex(i-1) = mostPositveWeightsIndex(i)
				} 
				mostPositveWeights(i) = weights(x)
				mostPositveWeightsIndex(i)=x
				i+=1
			}
			while(j < mostNegativeWeights.length && weights(x) < mostNegativeWeights(j)) {
				if (j > 0) {
					mostNegativeWeights(j-1) = mostNegativeWeights(j)
					mostNegativeWeightsIndex(j-1) = mostNegativeWeightsIndex(j)
				} 
				mostNegativeWeights(j) = weights(x)
				mostNegativeWeightsIndex(j)=x
				j+=1
			}
		} 
		for (j <- 0 until size) {
			val tmp1 = (vocabList(mostPositveWeightsIndex(j)), mostPositveWeights(j))
			val tmp2 = (vocabList(mostNegativeWeightsIndex(j)),mostNegativeWeights(j))
			mostPositive ::= tmp1
			mostNegative ::= tmp2
		}
		return (mostPositive,mostNegative)
	}
}