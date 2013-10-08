package pegasos

import math._
import scala.collection.mutable.Map
import scala.collection.mutable.HashMap

object Main {
	val trainingData = getDataFromFile("data/spam_train.txt")
	val testData = getDataFromFile("data/spam_test.txt")

	def main(args: Array[String]) {
		val start = System.currentTimeMillis
		var lambdas = List[Double]()
		for (i <- -9 to 1) {lambdas::= (math.pow(2,i))}
		lambdas = lambdas.reverse
		//lambdas.foreach(x => runPegasos(false,4000,x))
		runPegasos(true,5000,math.pow(2,-5))
		
		val end = System.currentTimeMillis

		println("Total Running Time of all Tests: " + (end-start)/1000.0 + " seconds")
	}

	def runPegasos(useTestData: Boolean,trainingSize:Int,lambda: Double) = {
		val start = System.currentTimeMillis
		val emailList = Support.getEmailList(trainingData)
		val tmp = emailList.splitAt(trainingSize) /*splits training data set for validation set*/
		val trainingDataSet = tmp._1
		val validationDataSet = if(useTestData) Support.getEmailList(testData) else tmp._2
		val vocabTally = Support.buildVocabulary(trainingDataSet)
		var vocabList = new Array[String](vocabTally.size)
		var vocabIndex = 0
		for(word <- vocabTally) { /*converts to simple array for faster iteration*/
			vocabList(vocabIndex) = word._1
			vocabIndex+=1
		}
		val trainingFeatureVectors = Support.makeFeatureVector(trainingDataSet,vocabList)
		val weights = Pegasos.pegasos_svm_train(trainingFeatureVectors,lambda)
		val trainingError = Pegasos.pegasos_svm_test(trainingFeatureVectors,weights)
		Pegasos.getSupportVectors(trainingFeatureVectors,weights)
		println(f"Average Training Error for lambda of $lambda is $trainingError%1.5f")
		val validationTrainingVectors = Support.makeFeatureVector(validationDataSet,vocabList)
		val testError = Pegasos.pegasos_svm_test(validationTrainingVectors,weights)
		println(f"Test Error: $testError%1.3f for lambda of $lambda")

		val algTime = System.currentTimeMillis

		println("\nAlgorithm Run Time: " + (algTime-start)/1000.0 + " seconds")
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