package classificationSc

import com.databricks.sparkdl.DeepImageFeaturizer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import utils.LabelByFilePathTf

object FlowersTruncCf {


  def main(args: Array[String]): Unit = {
    val sparky: SparkSession = FlowersCfConfig.sparkSession
    val flowersDir = "flowers/src/main/resources/flowers/*"

    // use image data source or binary file data source for images https://docs.databricks.com/data/data-sources/image.html
    val flowersDfAll: DataFrame = sparky.read.format("image").load(flowersDir)


    flowersDfAll.write.mode("overwrite").json("flowers/src/main/resources/flowersAsDs")

    val pipelineModel = loadLabelFeaturizeDs(flowersDfAll)

    // val tfDs = pipelineModel.transform(sparky.read.json("flowers/src/main/resources/flowersAsDs/part-00135-d200b7b8-6864-4c34-9c9b-353611413021-c000.json").as(Encoders.bean(ImageSchema.getClass)))
    //tfDs.printSchema()
    //tfDs.show(truncate = true)
    println("done")
  }


  def loadLabelFeaturizeDs(flowersDf: Dataset[Row]): PipelineModel = {
    val labelByFilePathTf: LabelByFilePathTf = new LabelByFilePathTf()

    labelByFilePathTf.setInputCol("image")
    labelByFilePathTf.setOutputCol("label")

    val featurizer: DeepImageFeaturizer = new DeepImageFeaturizer
    featurizer.setModelName("InceptionV3")
    featurizer.setInputCol("image")
    featurizer.setOutputCol("features")


    // works without classifier and pipeModel transform
    val pipe = new Pipeline()
    pipe.setStages(Array[PipelineStage](labelByFilePathTf, featurizer /*, getClassifier()*/))

    val pipeModel: PipelineModel = pipe.fit(flowersDf)

    pipeModel.write.overwrite().save("flowers/src/main/resources/cfModel")

    println("tranform ds")
    val tfFlowers = pipeModel.transform(flowersDf)

    println("showing transformed ds")
    tfFlowers.show()
    println("number of rows: " + tfFlowers.count())
    tfFlowers.printSchema()
    println("wrote albeled ds")

    val classifier:   LogisticRegression = getClassifier()

    println("fitting classifier")
    val cfModel = classifier.fit(tfFlowers)

    println("transforming with fitted classifier")
    val cfFlowers = cfModel.transform(tfFlowers)

    cfFlowers.show()
    cfFlowers.printSchema()

    pipeModel
  }

  private def getClassifier(): LogisticRegression = {

    val classifier = new LogisticRegression()
    classifier.setMaxIter(20)
    classifier.setRegParam(0.05)
    classifier.setElasticNetParam(0.3)
    classifier.setLabelCol("label")
    classifier.setFeaturesCol("features")

    classifier
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

  // cfModel.predict()
}
