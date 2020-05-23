package classificationSc

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.SparkSession

object FlowersCfConfig {


  private val sparkHome = null
  private val masterUri = "local"

  private val memory = "1200000000000"

  def sparkConf: SparkConf = {
    System.setProperty("hadoop.home.dir", "C:\\Users\\Cornelius\\hadoop-common-2.2.0-bin-master") // some dir to suppress warning
    System.setProperty("spark.driver.memory", "8g")



    val sparkConf = new SparkConf()
      .setAppName("flowers-mnist")
      .setMaster(masterUri)
      .set("spark.driver.memory", memory)
      .set("spark.testing.memory", memory)
      .set("spark.eventLog.enabled", "true")
      .set("spark.eventLog.dir", "flowers/spark-events")
      .set("spark.rdd.compress", "false")
      .set("spark.dynamicAllocation.enabled", "true")
      .set("spark.executor.memory", "16g") // this needs to be specified as a string
      .set("spark.executor.cores", "8") // this needs to be specified as a string
      .set("spark.executor.memoryOverhead", "8g") // this needs to be specified as a string
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .setSparkHome("/home/cornelius/.sdkman/candidates/spark/current")
   //   .set("spark.kryo.registrationRequired", "true")
    //     .set("--driver-memory", memory);
    sparkConf
  }

  def javaSparkContext: JavaSparkContext = { // shitty workaround because of the openjdk specifying java version as "10", while sparks wants three digits
    val javav = System.getProperty("java.version")

    if (javav.length == 2) System.setProperty("java.version", "1." + javav)

    val sparkContext =  new JavaSparkContext(sparkConf)

    sparkContext.setCheckpointDir("flowers/src/main/resources/checkpoints")

    sparkContext
  }

  def sparkSession: SparkSession = {

    SparkSession.builder().config(sparkConf).getOrCreate()

    // SparkSession.builder.sparkContext(javaSparkContext.sc).appName("Fashion MNIST").getOrCreate
  }


}
