package classificationSc

import com.databricks.sparkdl.DeepImageFeaturizer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.mlflow.api.proto.Service.RunInfo
import org.mlflow.tracking.MlflowClient
import utils.LabelByFilePathTf

object Jpeg2DsConverter {


  def main(args: Array[String]): Unit = {

    println("converting images to Spark dataset")


    val mlflowClient: MlflowClient = new MlflowClient("http://localhost:5000")

    // the id is arbitrary, and has to be a numeric-char-only String val
    val experimentId = "1" // mlflowClient.createExperiment("flowers classification")

    println("experiment id: " + experimentId)

    val runInfo: RunInfo = mlflowClient.createRun(experimentId)

    val sparky: SparkSession = FlowersCfConfig.sparkSession
    val flowersDir = "flowers/src/main/resources/flowers/*"

    /* IMPORTANT NOTE:
      Check if input directory only contains .jpg files (or other image file endings)
      There is currently no automatic filtering for things like .py, .txt
     */

    // use image data source or binary file data source for images https://docs.databricks.com/data/data-sources/image.html
    val flowersDf: DataFrame = sparky.read.format("image").load(flowersDir)


    //flowersDf.write.mode("overwrite").json("flowers/src/main/resources/flowersAsDs")

    //println("number of rows: "+ flowersDfAll.count())
    //flowersDfAll.cache()
    // take first for debug purposes
    //val flowersDf = flowersDfAll.limit(800)

    sparky.sparkContext.setCheckpointDir("flowers/src/main/resources/checkpoints")
    //flowersDf.checkpoint(true)
    //flowersDf.write.mode("overwrite").json("flowers/src/main/resources/flowersTruncated")

    val labelByFilePathTf: LabelByFilePathTf = new LabelByFilePathTf()

    labelByFilePathTf.setInputCol("image")
    labelByFilePathTf.setOutputCol("label")

    val flowersDfLabeled = labelByFilePathTf.transform(flowersDf)

    flowersDfLabeled.show(truncate = true)
    flowersDfLabeled.printSchema()

    val featurizer: DeepImageFeaturizer = new DeepImageFeaturizer
    featurizer.setModelName("InceptionV3")
    featurizer.setInputCol("image")
    featurizer.setOutputCol("features")


    //   val pipe = new Pipeline()
    //    pipe.setStages(Array[PipelineStage](featurizer, getClassifier()))
    val flowersDfLabeledFeaturized = featurizer.transform(flowersDfLabeled)
    flowersDfLabeledFeaturized.write.mode("overwrite") json ("flowers/src/main/resources/flowersAsDs")

    val trainingAndTestDFs = flowersDfLabeledFeaturized.randomSplit(Array[Double](0.4, 0.6))

    val trainingSet = trainingAndTestDFs(1)
    //trainingSet.cache()
    val testSet = trainingAndTestDFs(0)
   // testSet.cache()

    println("training set:")
    trainingSet.show()
    trainingSet.printSchema()

    println("test set:")
    testSet.show()
    testSet.printSchema()

    val featuresCols = trainingSet.select("features")
    featuresCols.show(50)
    featuresCols.printSchema()

    val testSetCf = classifyDf(trainingSet, testSet)
    evaluateDf(testSetCf)

    mlflowClient.setTerminated(runInfo.getRunId)
    mlflowClient.logMetric(runInfo.getRunId, "number_of_rows", 111)


    //mlflowClient.logArtifact(runInfo.getRunId, new File("flowers/src/main/resources/cfModel"))
  }


  private def classifyDf(trainingSet: Dataset[Row], testSet: Dataset[Row]) = {
    val classifier = new LogisticRegression()
    classifier.setMaxIter(20)
    classifier.setRegParam(0.05)
    classifier.setElasticNetParam(0.3)
    classifier.setLabelCol("label")
    //    classifier.setFeaturesCol("features")

    val checkpointSet = trainingSet.checkpoint(true)
    val model = classifier.fit(checkpointSet)

    model.write.overwrite().save("flowers/src/main/resources/cfModel")
    println(model.summary)
    val testSetCf = model.transform(testSet)
    testSetCf.show()
    testSetCf.printSchema()
    testSetCf
  }

  private def evaluateDf(fashionDsTf: DataFrame) = {
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()

    println("evaluating")
    val evaluationResult = evaluator.evaluate(fashionDsTf)
    println("evaluation result: " + evaluationResult + " larger is better: " + evaluator.isLargerBetter)
  }
}


