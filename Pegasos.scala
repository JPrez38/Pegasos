package pegasos

import math._
import scala.collection.mutable.Map
import scala.collection.mutable.HashMap

object Pegasos {
	def dot[T <% Double](m1: Iterable[T], m2: Iterable[Double]) : Double = {
		require(m1.size == m2.size) 
		return (for ((x, y) <- m1 zip m2) yield x * y).sum
	}

	def magnitude(x:Array[Double]) : Double = {
		var magnitude = math.sqrt((for (x_i <- x) yield x_i*x_i).sum)
		if (magnitude == 0.0) return .000001 else magnitude
	}

	def subtractVector(a:Array[Double],b:Array[Double]) : Array[Double] = {
		var c = new Array[Double](a.size)
		require(a.size == b.size)
		for (i <- 0 to a.size-1){ c(i) = (a(i) - b(i)) }
		return c
	}

	def addVector(a:Array[Double],b:Array[Double]) : Array[Double] = {
		var c = new Array[Double](a.size)
		require(a.size == b.size)
		for (i <- 0 to a.size-1){ c(i) = (a(i) + b(i)) }
		return c
	}

	def scalarVectorMultiply(a: Array[Double],scalar:Double) : Array[Double] = {
		var b = new Array[Double](a.size)
		for (i <- 0 to a.size-1) { b(i) = a(i) * scalar}
		return b	
	}

	def scalarVectorMultiply(a: Array[Int],scalar:Double) : Array[Double] = {
		var b = new Array[Double](a.size)
		for (i <- 0 to a.size-1) { b(i) = a(i) * scalar}
		return b	
	}

	def pegasos_svm_train(data: List[(Array[Int],Int)],lambda: Double) : Array[Double] = {
		var weights = new Array[Double](data(0)._1.size)
		var del = new Array[Double](weights.size)
		var t = 0
		var sumHingeLoss = 0.0
		for (iter <- 1 to 20) {
			var empiricalLoss = 0.0
			var misclassified = 0.0
			for (email <- data) {
				val x = email._1
				val y = email._2
				t += 1
				val eta = 1/(t*lambda)
				if ((y*dot(x,weights) < 1)) {
					del = addVector(scalarVectorMultiply(weights,(1-(eta*lambda))),scalarVectorMultiply(x,(eta*y)))
				} else {
					del = scalarVectorMultiply(weights,(1-eta*lambda))
				}
				val tmp = (1/(math.sqrt(lambda)))/magnitude(del)
				weights = scalarVectorMultiply(del,math.min(1,tmp))
			}
			for (email <- data) {
				val x = email._1
				val y = email._2
				val out = if (dot(x,weights) > 0) 1 else -1
				empiricalLoss = empiricalLoss + math.max(0,1-out*y)
			}
			empiricalLoss = empiricalLoss / data.size /* also the hinge loss */
			sumHingeLoss += empiricalLoss
			val regularizationTerm = (lambda * math.pow(magnitude(weights),2)) / 2
			val trainingError = misclassified / data.size
			val svmObjective = regularizationTerm + empiricalLoss
			println(f"SVM Objective at iteration $iter is: $svmObjective%1.3f")
		}
		val avgHingeLoss = sumHingeLoss / 20
		println(f"Average Hinge Loss for lambda $lambda is: $avgHingeLoss%1.4f")

		return weights
	}

	def getSupportVectors(data: List[(Array[Int],Int)],weights: Array[Double]) = {
		var supportVectors=0
		for (email <- data) {
			val dotted = math.abs(dot(email._1,weights)) 
			if (dotted > .99 && dotted < 1.01) { supportVectors += 1} /*accounts for minor floating point rounding errors */	
		}

		println("Number of support vectors: " + supportVectors)
	}

	def pegasos_svm_test(data: List[(Array[Int],Int)],weights: Array[Double]) : Double = {
		var errorCount=0
		for( x <- data) {
			val featureVector = x._1
			var desiredOutput = x._2

			val output = if (dot(featureVector,weights) > 0) 1 else -1
			val error = desiredOutput-output
			if (error != 0) errorCount+=1
		}
		return errorCount/data.size.toDouble
	}
}