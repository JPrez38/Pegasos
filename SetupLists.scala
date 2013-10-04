package pegasos

import math._
import scala.collection.mutable.Map
import scala.collection.mutable.HashMap

object Support {
	def getEmailList(inputFile: String) : List[(Int,String)] = {
		var emails = List[(Int,String)]() 
		var newIndex = 0
		var currentIndex = 0
		def constructEmail(beginIndex: Int, endIndex: Int, input: String) : (Int,String) = {
			val emailText = input.slice(beginIndex,endIndex)
			val classification = if (emailText.head.asDigit == 1) 1 else -1
			return (classification,emailText.substring(1,emailText.length))
		}
		def setIndex(nextIndex: Int, defaultValue: Int) : Int = {
			return (if (nextIndex > -1) nextIndex else defaultValue)
		}
		while({
			newIndex = math.min(setIndex(inputFile.indexOf("0",currentIndex+1),inputFile.length+1),
			setIndex(inputFile.indexOf("1",currentIndex+1),inputFile.length+1));
			newIndex != inputFile.length+1}) { /*iterates over String finding all 1's or 0's */

			val email = constructEmail(currentIndex,newIndex,inputFile)
			emails ::= email
			currentIndex=newIndex
		}
		val email = constructEmail(currentIndex,inputFile.length,inputFile)
		emails ::= email /* scala notation for adding item to begining of list */
		return emails.reverse /* reverses to get correct ordering */
	}

	def buildVocabulary(emailList: List[(Int,String)]) : Map[String,Int] = {
		var vocabulary = Map[String,Int]()
		for (email <- emailList) {
			val words = email._2.split(" ")
			var wordList = Set[String]()
			words.foreach(word => wordList += word)
			for (word <- wordList) {
				var count = vocabulary.getOrElse(word,0)
				count += 1
				vocabulary += word -> count
			}
		}
		vocabulary.remove("") /* removes a random no space item from vocab list */
		return vocabulary.retain((k,v) => v >= 30) /* returns a map of items that have value greater than 30 */
	} 

	def makeFeatureVector(emailList: List[(Int,String)], vocabList: Array[String]) : List[(Array[Int],Int)] = {
		var emailFeatureVector = List[(Array[Int],Int)]()
		var i = 0
		for (email <- emailList) {
			var featureVector = new Array[Int](vocabList.size)
			var vocabIndex = 0
			val words = email._2.split(" ")
			var wordList = HashMap[String,Int]()
			words.foreach(word => wordList += word -> 0) 
			for (vocabWord <- vocabList) {
				featureVector(vocabIndex) = if(wordList.getOrElse(vocabWord,None) != None) 1 else -1
				vocabIndex+=1
			}
			val vector = (featureVector,email._1)
			emailFeatureVector ::= vector
			i+=1
		}
		return emailFeatureVector.reverse
	}
}