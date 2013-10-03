package pegasos

import math._
import scala.collection.mutable.Map
import scala.collection.mutable.HashMap

object Pegasos {
	def dot[T <% Double](m1: Iterable[Int], m2: Iterable[Double]) : Int = {
		require(m1.size == m2.size) 
		if (((for ((x, y) <- m1 zip m2) yield x * y).sum) >= .5) return 1 else 0
	}

	def dotted[T <% Double](m1: Iterable[Int], m2: Iterable[Double]) : Double = {
		require(m1.size == m2.size) 
		return (for ((x, y) <- m1 zip m2) yield x * y).sum
	}

	def magnitude(x:Array[Double]) : Double = {
		var magnitude = math.sqrt((for (x_i <- x) yield x_i*x_i).sum)
		if (magnitude == 0.0) return 0.1 else magnitude
	}

	def subtractVector(a:Array[Double],b:Array[Double]) : Array[Double] = {
		var c = new Array[Double](a.size)
		require(a.size == b.size)
		for (i <- 0 to a.size-1){ c(i) = (a(i) - b(i)) }
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
		for (iter <- 1 to 20) {
			for (email <- data) {
				val x = email._1
				val y = email._2
				t += 1
				val eta = 1/(t*lambda)
				if ((y*dotted(x,weights) < 1)) {
					del = subtractVector(scalarVectorMultiply(weights,(1-eta)),scalarVectorMultiply(x,(eta*y)))
				} else {
					del = scalarVectorMultiply(weights,(1-eta))
				}
				weights = scalarVectorMultiply(del,math.min(1,(1/(math.sqrt(lambda)))/magnitude(del)))
			}
		}
		return weights
	}

	def pegasos_svm_test(data: List[(Array[Int],Int)],weights: Array[Double]) : Double = {
		var errorCount=0
		for( x <- data) {
			val featureVector = x._1
			val desiredOutput = x._2

			val output = dot(featureVector,weights)
			//println(output)
			val error = desiredOutput-output
			if (error != 0) errorCount+=1
		}
		return errorCount/data.size.toDouble
	}
}