package utils

import java.util

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DataType, DataTypes}

class LabelByFilePathTf(override val uid: String)
  extends UnaryTransformer[Row, Int, LabelByFilePathTf] with DefaultParamsWritable {

  private val labelsAsString: util.HashSet[String] = new util.HashSet[String]()


  def this() {


    this(
      Identifiable.randomUID("label_by_file_path_tf")
    )
  }

  override protected def createTransformFunc: Row => Int = (image: Row) => {
    val imagePath: String = image.getAs(0)

    val pathSplit: Array[String]  = imagePath.split("/")

    // TODO use dynamic Hashmap to set labels
    pathSplit(pathSplit.length-2) match {
      case "tulip" => 0
      case "daisy" => 1
      case "rose" => 2
      case "sunflower" => 3
      case "dandelion" => 4
      case _ => 5
    }

  }


  override protected def outputDataType: DataType = {

    DataTypes.IntegerType
  }
}
