package classificationSc

import com.databricks.sparkdl.DeepImageFeaturizer
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame

object ApplyModelCf {


  def main(args: Array[String]): Unit = {

    val sparkSession = FlowersCfConfig.sparkSession

    val cfModel = LogisticRegressionModel.load("flowers/src/main/resources/cfModel")

    val daisyDir = "flowers/src/main/resources/singleDaisy"

    // use image data source or binary file data source for images https://docs.databricks.com/data/data-sources/image.html
    val daisyDf: DataFrame = sparkSession.read.format("image").load(daisyDir)

    val singleRowDs = daisyDf.limit(1)

    val featurizer: DeepImageFeaturizer = new DeepImageFeaturizer
    featurizer.setModelName("InceptionV3")
    featurizer.setInputCol("image")
    featurizer.setOutputCol("features")

    //val rfModel =   RandomForestClassificationModel.load("")


    // some vals to avoid the unreadable oneliner
    val tfDs = featurizer.transform(singleRowDs)
    val rows = tfDs.collect()
    val row = rows(0)
    val featureVector: DenseVector = row.getAs("features")

    val prediction = cfModel.predict(featureVector)
    val probability: DenseVector = cfModel.transform(tfDs).select("probability").collect()(0).getAs("probability")

    println("probabilities: " + probability)
    println("prediction: " + prediction)
  }

}
