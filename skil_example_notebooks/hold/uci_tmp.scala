import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.{DenseLayer, GravesLSTM, OutputLayer, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions


import org.apache.commons.io.IOUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.java.function.FlatMapFunction
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.writable.Writable
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.api.Repartition
import org.deeplearning4j.spark.api.RepartitionStrategy
import org.deeplearning4j.spark.api.TrainingMaster
import org.deeplearning4j.spark.api.stats.SparkTrainingStats
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.spark.stats.StatsUtils
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
//import org.nd4j.linalg.dataset.api.preprocessor.SparkDataNormalization
//import org.nd4j.linalg.dataset.api.preprocessor.SparkNormalizerStandardize
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.INDArrayIndex
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.Logger
import org.slf4j.LoggerFactory

import java.io.File
import java.io.FileWriter
import java.io.PrintStream
import java.io.Serializable
import java.net.URL
import java.util._

import scala.beans.{BeanProperty, BooleanBeanProperty}
import scala.collection.JavaConversions._


/*
trait SparkDataSetPreProcessor extends Serializable {

  def preProcess(toPreProcess: JavaRDD[DataSet]): JavaRDD[DataSet]

}
*/
import org.apache.spark.rdd._
import org.apache.spark.rdd.RDD
//import org.apache.spark.rdd.RDD
//import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.DataSet

trait SparkDataSetPreProcessor extends Serializable {

    import org.apache.spark.rdd.RDD
    import org.nd4j.linalg.dataset.DataSet


  def preProcess(toPreProcess: RDD[DataSet]): RDD[DataSet]
   // def preProcess(toPreProcess: RDD[String]): RDD[String]
  //def preProcess(toPreProcess: DataSet): DataSet

}

trait SparkDataNormalization extends SparkDataSetPreProcessor {

    import org.apache.spark.rdd.RDD
    import org.nd4j.linalg.dataset.DataSet

  def fit(dataRDD: RDD[DataSet]): Unit

}


class SparkNormalizerStandardize extends SparkDataNormalization {

    import org.nd4j.linalg.api.ndarray.INDArray
    import org.apache.spark.rdd.RDD
    import org.nd4j.linalg.dataset.DataSet
    import org.apache.spark.api.java.function.FlatMapFunction
    
    import java.io.File
    import java.io.FileWriter
    import java.io.PrintStream
    import java.io.Serializable
    import java.net.URL
    import java.util._
    
    import scala.beans.{BeanProperty, BooleanBeanProperty}
    import scala.collection.JavaConversions._


  private var featureMean: INDArray = _

  private var featureStd: INDArray = _

  override def preProcess(toPreProcess: RDD[DataSet]): RDD[DataSet] = {
    val mean: INDArray = this.featureMean
    val std: INDArray = this.featureStd
    if (this.featureMean == null) {
      throw new RuntimeException(
        "API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)")
    } else {
      toPreProcess.mapPartitions(
        new FlatMapFunction[Iterator[DataSet], DataSet]() {
          override def call(
              iter: Iterator[DataSet]): java.lang.Iterable[DataSet] = {
            val out: List[DataSet] = new ArrayList[DataSet]()
            while (iter.hasNext) {
              val ds: DataSet = iter.next()
              if (ds.getFeatures.rank() == 2) {
                ds.getFeatures.subi(mean)
                ds.getFeatures.divi(std)
                out.add(ds)
              } else {
                System.err.println(
                  ds.getFeatures
                    .rank() + " rank preProcessor should be implemented")
              }
            }
            out
          }
        })
    }
  }

  override def fit(dataRDD: RDD[DataSet]): Unit = {
    val doubleRDD: JavaDoubleRDD =
      dataRDD.flatMapToDouble(new DoubleFlatMapFunction[DataSet]() {
        override def call(dataSet: DataSet): java.lang.Iterable[Double] = {
          var splited: Array[String] = dataSet.getFeatures.toString
            .replace("[", "")
            .replace("]", "")
            .split(", ")
          var converted: List[Double] = new ArrayList[Double]()
          for (s <- splited) {
            converted.add(java.lang.Double.parseDouble(s))
          }
          converted
        }
      })
    val stats: StatCounter = doubleRDD.stats()
    featureMean = Nd4j.create(1, 1)
    featureMean.putScalar(Array(0, 0), stats.mean())
    featureStd = Nd4j.create(1, 1)
    featureStd.putScalar(Array(0, 0), stats.stdev())
  }

  def getFeatureMean(): INDArray =
    if (this.featureMean == null) {
      throw new RuntimeException(
        "API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)")
    } else {
      featureMean
    }

  def getFeatureStd(): INDArray =
    if (this.featureStd == null) {
      throw new RuntimeException(
        "API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)")
    } else {
      featureStd
    }

}

//remove if not needed

class ModelWithEval(@BeanProperty var model: MultiLayerNetwork,
                    @BeanProperty var eval: Evaluation,
                    @BeanProperty var testEvals: Array[Evaluation])


private class BatchDataSetsFunction(private val minibatchSize: Int)
    extends FlatMapFunction[Iterator[DataSet], DataSet] {

  override def call(iter: Iterator[DataSet]): java.lang.Iterable[DataSet] = {
    val out: List[DataSet] = new ArrayList[DataSet]()
    while (iter.hasNext) {
      val list: List[DataSet] = new ArrayList[DataSet]()
      var count: Int = 0
      while (count < minibatchSize && iter.hasNext) {
        val ds: DataSet = iter.next()
        count += ds.getFeatureMatrix.size(0)
        list.add(ds)
      }
      var next: DataSet = null
      next = if (list.size == 0) list.get(0) else DataSet.merge(list)
      out.add(next)
    }
    val reshpaedOut: List[DataSet] = new ArrayList[DataSet]()
    for (i <- 0 until out.size) {
      val ds: DataSet = out.get(i)
      val features: INDArray = ds.getFeatures
      val labels: INDArray = ds.getLabels
      val featureShape: Array[Int] = features.shape()
      val labelShape: Array[Int] = labels.shape()
      ds.setFeatures(features.reshape(featureShape(0), 1, featureShape(1)))
      ds.setLabels(labels.reshape(labelShape(0), 1, labelShape(1)))
      reshpaedOut.add(converToSequenceDataFormat(ds))
    }
    reshpaedOut
  }

  private def converToSequenceDataFormat(d: DataSet): DataSet = {
    val nDataSets: Int = d.numExamples()
    val featureList: List[INDArray] = new ArrayList[INDArray](nDataSets)
    val labelList: List[INDArray] = new ArrayList[INDArray](nDataSets)
    for (i <- 0 until nDataSets) {
      val features: INDArray = d.get(i).getFeatures.reshape(60, 1)
//2d time series, with shape [timeSeriesLength,vectorSize]
      val labels: INDArray = d.get(i).getLabels
      featureList.add(features)
      labelList.add(labels)
    }
//Convert 2d sequences/time series to 3d minibatch data
    var featuresOut: INDArray = null
    var labelsOut: INDArray = null
    var featuresMask: INDArray = null
    var labelsMask: INDArray = null
    var longestTimeSeries: Int = 0
    for (features <- featureList) {
      longestTimeSeries = Math.max(features.size(0), longestTimeSeries)
    }
    for (labels <- labelList) {
      longestTimeSeries = Math.max(labels.size(0), longestTimeSeries)
    }
    val featuresShape: Array[Int] = Array( //# examples
      featureList.size, //example vector size
      featureList.get(0).size(1),
      longestTimeSeries)
    val labelsShape: Array[Int] = Array( //# examples
                                        labelList.size, //example vector size
                                        labelList.get(0).size(1),
                                        longestTimeSeries)
    featuresOut = Nd4j.create(featuresShape, 'f')
    labelsOut = Nd4j.create(labelsShape, 'f')
    featuresMask = Nd4j.ones(featureList.size, longestTimeSeries)
    labelsMask = Nd4j.ones(labelList.size, longestTimeSeries)
    for (i <- 0 until featureList.size) {
      val f: INDArray = featureList.get(i)
      val l: INDArray = labelList.get(i)
      val fLen: Int = f.size(0)
      val lLen: Int = l.size(0)
      if (fLen >= lLen) {
//Align labels with end of features (features are longer)
        featuresOut
          .tensorAlongDimension(i, 1, 2)
          .permutei(1, 0)
          .put(Array(NDArrayIndex.interval(0, fLen), NDArrayIndex.all()), f)
        labelsOut
          .tensorAlongDimension(i, 1, 2)
          .permutei(1, 0)
          .put(Array(NDArrayIndex.interval(fLen - lLen, fLen),
                     NDArrayIndex.all()),
               l)
        for (j <- fLen until longestTimeSeries) {
          featuresMask.putScalar(i, j, 0.0)
        }
        for (j <- 0 until fLen - lLen) {
          labelsMask.putScalar(i, j, 0.0)
        }
        for (j <- fLen until longestTimeSeries) {
          labelsMask.putScalar(i, j, 0.0)
        }
      } else {
//Align features with end of labels (labels are longer)
        featuresOut
          .tensorAlongDimension(i, 1, 2)
          .permutei(1, 0)
          .put(Array(NDArrayIndex.interval(lLen - fLen, lLen),
                     NDArrayIndex.all()),
               f)
        labelsOut
          .tensorAlongDimension(i, 1, 2)
          .permutei(1, 0)
          .put(Array(NDArrayIndex.interval(0, lLen), NDArrayIndex.all()), l)
        for (j <- 0 until lLen - fLen) {
          featuresMask.putScalar(i, j, 0.0)
        }
        for (j <- lLen until longestTimeSeries) {
          featuresMask.putScalar(i, j, 0.0)
        }
        for (j <- lLen until longestTimeSeries) {
          labelsMask.putScalar(i, j, 0.0)
        }
      }
    }
    val ds: DataSet =
      new DataSet(featuresOut, labelsOut, featuresMask, labelsMask)
    ds
  }

}     

  private def downloadUCIData(): Unit = {
    var isHDFS: Boolean = false
    var exist: Boolean = false
    if (dataPath.startsWith("hdfs:") || dataPath.startsWith("wasb:")) {
      isHDFS = true
    }
    if (isHDFS) {
// if datapath is hdfs, check hdfs path
      val configuration: Configuration = new Configuration()
      val hdfs: FileSystem = FileSystem.get(configuration)
      exist = hdfs.exists(new Path(dataPath))
// save data to tempDir
      baseDir = new File(System.getProperty("java.io.tmpdir"))
    } else {
// if datapath is local, check local path
      baseDir = new File(dataPath)
      exist = baseDir.exists()
    }
    if (exist) {
      println(dataPath + " exist!!")
//Data already exists, don't download it again
      return
    }
    baseTrainDir = new File(baseDir, "train")
    baseTestDir = new File(baseDir, "test")
    val url: String =
      "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data"
    val data: String = IOUtils.toString(new URL(url))
    val lines: Array[String] = data.split("\n")
//Create directories
    baseDir.mkdir()
    baseTrainDir.mkdir()
    baseTestDir.mkdir()
    var lineCount: Int = 0
    val rawData: List[String] = new ArrayList[String]()
    for (line <- lines) {
      val transposed: String = line.replaceAll(" +", " ")
//Labels: first 100 examples (lines) are label 0, second 100 examples are label 1, and so on
      rawData.add(
        transposed + " " + ({ lineCount += 1; lineCount - 1 } / 100) +
          "\n")
    }
//Randomize and do a train/test split:
    Collections.shuffle(rawData, new Random(12345))
    val outTrain: File = new File(baseTrainDir, "train.csv")
    val outTest: File = new File(baseTestDir, "test.csv")
//75% train, 25% test
    val nTrain: Int = 450
    val trainData: List[String] = rawData.subList(0, nTrain)
    val testData: List[String] = rawData.subList(nTrain, rawData.size)
    var fw: FileWriter = new FileWriter(outTrain)
    for (d <- trainData) {
      fw.write(d)
    }
    fw.close()
    fw = new FileWriter(outTest)
    for (d <- testData) {
      fw.write(d)
    }
    fw.close()
    if (isHDFS || copyDataToHDFS) {
      val userName: String = System.getProperty("user.name")
      val configuration: Configuration = new Configuration()
      val hdfs: FileSystem = FileSystem.get(configuration)
      val trPath: Path = new Path(
        "hdfs:///user/" + userName + "/UCIsequence/train")
      val tePath: Path = new Path(
        "hdfs:///user/" + userName + "/UCIsequence/test")
      hdfs.mkdirs(trPath)
      hdfs.mkdirs(tePath)
      hdfs.copyFromLocalFile(false, true, new Path(outTrain.toString), trPath)
      hdfs.copyFromLocalFile(false, true, new Path(outTest.toString), tePath)
    }
    if (isHDFS) {
      outTrain.delete()
      outTest.delete()
      baseTrainDir.delete()
      baseTestDir.delete()
      baseDir.delete()
    }
  }


	// start main training run setup
    val useSparkLocal: Boolean = true
    val miniBatchSize: Int = 10
    val nEpochs: Int = 40
    val averagingFrequency: Int = 10
    val learningRate: Double = 0.05
    val dataPath: String = "src/main/resources/uci/"
    val writeStats: Boolean = false
    val copyDataToHDFS: Boolean = false
    val saveUpdater: Boolean = true
    val modelPath: String = String.format("/tmp/%s-trained-%d.zip",
                                          this.getClass.getSimpleName,
                                          System.currentTimeMillis())
    val sparkClose: Boolean = false
    val help: Boolean = false
    val baseDir: File = null
    val baseTrainDir: File = null
    val baseTestDir: File = null
    val out: PrintStream = System.out
    val err: PrintStream = System.err








// code block


        //=====================================================================
        //                Step 0: Prepare data
        //=====================================================================
        downloadUCIData();

        //=====================================================================
        //      Step 1: Load data and execute the operations on Spark
        //=====================================================================
/*
        val sparkConf = new SparkConf();
        if (useSparkLocal) sparkConf.setMaster("local[*]");
        sparkConf.setAppName("UCI Sequence Classificiation Spark Job");
        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");

        if (sc == null) {
          sc = new JavaSparkContext(sparkConf);
        }
*/
// load data
    val linesTR: JavaRDD[String] = sc.textFile(dataPath + "/train/train.csv")
    val linesTE: JavaRDD[String] = sc.textFile(dataPath + "/test/test.csv")
// prepare rdd
    val writablesTR: JavaRDD[List[Writable]] =
      linesTR.map(new StringToWritablesFunction(new CSVRecordReader(0, ",")))
    val dataSetsTR: JavaRDD[DataSet] =
      writablesTR.map(new DataVecDataSetFunction(60, 60, 6, false, null, null))
    val writablesTE: JavaRDD[List[Writable]] =
      linesTE.map(new StringToWritablesFunction(new CSVRecordReader(0, ",")))
    val dataSetsTE: JavaRDD[DataSet] =
      writablesTE.map(new DataVecDataSetFunction(60, 60, 6, false, null, null))
// normalize
    val normalization: SparkDataNormalization =
      new SparkNormalizerStandardize()
    normalization.fit(dataSetsTR)
// apply normalization of train set to both tr and te
    val trainRDDNorm: JavaRDD[DataSet] = normalization.preProcess(dataSetsTR)
    val testRDDNorm: JavaRDD[DataSet] = normalization.preProcess(dataSetsTE)
// create batch dataset
    val trainRDD: JavaRDD[DataSet] =
      trainRDDNorm.mapPartitions(new BatchDataSetsFunction(miniBatchSize))
    val testRDD: JavaRDD[DataSet] =
      testRDDNorm.mapPartitions(new BatchDataSetsFunction(miniBatchSize))








    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(123L)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(learningRate, 0.9))
      .gradientNormalization( //Not always required, but helps with this data set
        GradientNormalization.ClipElementWiseAbsoluteValue)
      .gradientNormalizationThreshold(0.5)
      .list()
      .layer(0,
             new GravesLSTM.Builder()
               .activation(Activation.TANH)
               .nIn(1)
               .nOut(10)
               .build())
      .layer(1,
             new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
               .activation(Activation.SOFTMAX)
               .nIn(10)
               .nOut(6)
               .build())
      .pretrain(false)
      .backprop(true)
      .build()



//Configuration for Spark training: see http://deeplearning4j.org/spark for explanation of these configuration options
    val tm: TrainingMaster =
      new ParameterAveragingTrainingMaster.Builder(miniBatchSize)
        .averagingFrequency(averagingFrequency)
        .saveUpdater(saveUpdater)
        .workerPrefetchNumBatches(2)
        .batchSizePerWorker(miniBatchSize)
        .repartionData(Repartition.Always)
        .repartitionStrategy(RepartitionStrategy.SparkDefault)
        .build()

    val sparkNet: SparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc, conf, tm)
    sparkNet.setCollectTrainingStats(true)
    
    val listeners: ArrayList[IterationListener] =
      new ArrayList[IterationListener]()
    
    listeners.add(new ScoreIterationListener(1))
    sparkNet.setListeners(listeners)
    
    var model: MultiLayerNetwork = null
    val testEvals: ArrayList[Evaluation] = new ArrayList[Evaluation]()
    val str: String =
      "%s set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f"
    
    for (i <- 0 until nEpochs) {
      model = sparkNet.fit(trainRDD)
      val evaluationt: Evaluation = sparkNet.evaluate(trainRDD)
      println(
        String.format(str, "tr", i, evaluationt.accuracy(), evaluationt.f1()))
      val evaluation: Evaluation = sparkNet.evaluate(testRDD)
      println(
        String.format(str, "te", i, evaluation.accuracy(), evaluation.f1()))
      testEvals.add(evaluation)
    }
    if (writeStats) {
      val stats: SparkTrainingStats = sparkNet.getSparkTrainingStats
      StatsUtils.exportStatsAsHtml(
        stats,
        "SparkStats_" + System.currentTimeMillis() + ".html",
        sc)
    }
    println("----- Example Complete -----")
    val locationToSave: File = new File(modelPath)
    ModelSerializer.writeModel(model, locationToSave, saveUpdater)
    printf("---- Saved model to: %s -----\n", locationToSave)
    val finalEval: Evaluation = sparkNet.evaluate(testRDD)
    val result: ModelWithEval = new ModelWithEval(
      model,
      finalEval,
      testEvals.toArray(Array.ofDim[Evaluation](testEvals.size)))

