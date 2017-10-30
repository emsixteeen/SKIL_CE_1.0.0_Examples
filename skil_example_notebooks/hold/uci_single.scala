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


import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize


import java.io.File
import java.net.URL
import java.util.ArrayList
import java.util.Collections
import java.util.List
import java.util.Random


// SETUP variables ---------------------- 

    val baseDir: File = new File("src/main/resources/uci/")
    val baseTrainDir: File = new File(baseDir, "train")
    val featuresDirTrain: File = new File(baseTrainDir, "features")
    val labelsDirTrain: File = new File(baseTrainDir, "labels")
    val baseTestDir: File = new File(baseDir, "test")
    val featuresDirTest: File = new File(baseTestDir, "features")
    val labelsDirTest: File = new File(baseTestDir, "labels")



// download dataset function -----


  def downloadUCIData() {
//Data already exists
    if (baseDir.exists()) return
    val url: String =
      "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data"
    val data: String = IOUtils.toString(new URL(url))
    val lines: Array[String] = data.split("\n")
    if (baseDir.exists()) baseDir.delete()
    baseDir.mkdir()
    baseTrainDir.mkdir()
    featuresDirTrain.mkdir()
    labelsDirTrain.mkdir()
    baseTestDir.mkdir()
    featuresDirTest.mkdir()
    labelsDirTest.mkdir()
    var lineCount: Int = 0
    val contentAndLabels: List[Pair[String, Integer]] =
      new ArrayList[Pair[String, Integer]]()
    for (line <- lines) {
      val transposed: String = line.replaceAll(" +", "\n")
//Labels: first 100 are label 0, second 100 are label 1, and so on
      contentAndLabels.add(new Pair(transposed, {
        lineCount += 1; lineCount - 1
      } / 100))
    }
//Randomize and do a train/test split:
    Collections.shuffle(contentAndLabels, new Random(12345))
//75% train, 25% test
    val nTrain: Int = 450
    var trainCount: Int = 0
    var testCount: Int = 0
    for (p <- contentAndLabels) {
      var outPathFeatures: File = null
      var outPathLabels: File = null
      if (trainCount < nTrain) {
        outPathFeatures = new File(featuresDirTrain, trainCount + ".csv")
        outPathLabels = new File(labelsDirTrain, trainCount + ".csv") {
          trainCount += 1; trainCount - 1
        }
      } else {
        outPathFeatures = new File(featuresDirTest, testCount + ".csv")
        outPathLabels = new File(labelsDirTest, testCount + ".csv") {
          testCount += 1; testCount - 1
        }
      }
      FileUtils.writeStringToFile(outPathFeatures, p.getFirst)
      FileUtils.writeStringToFile(outPathLabels, p.getSecond.toString)
    }
  }




// do basic ETL setup -----------



    downloadUCIData()
// ----- Load the training data -----
    val trainFeatures: SequenceRecordReader = new CSVSequenceRecordReader()
    trainFeatures.initialize(
      new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath + "/%d.csv",
                                 0,
                                 449))
    val trainLabels: SequenceRecordReader = new CSVSequenceRecordReader()
    trainLabels.initialize(
      new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath + "/%d.csv",
                                 0,
                                 449))
    val minibatch: Int = 10
    val numLabelClasses: Int = 6
    val trainData: DataSetIterator = new SequenceRecordReaderDataSetIterator(
      trainFeatures,
      trainLabels,
      minibatch,
      numLabelClasses,
      false,
      SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)
//Normalize the training data
    val normalizer: DataNormalization = new NormalizerStandardize()
//Collect training data statistics
    normalizer.fit(trainData)
    trainData.reset()
//Use previously collected statistics to normalize on-the-fly
    trainData.setPreProcessor(normalizer)
// ----- Load the test data -----
    val testFeatures: SequenceRecordReader = new CSVSequenceRecordReader()
    testFeatures.initialize(
      new NumberedFileInputSplit(featuresDirTest.getAbsolutePath + "/%d.csv",
                                 0,
                                 149))
    val testLabels: SequenceRecordReader = new CSVSequenceRecordReader()
    testLabels.initialize(
      new NumberedFileInputSplit(labelsDirTest.getAbsolutePath + "/%d.csv",
                                 0,
                                 149))
    val testData: DataSetIterator = new SequenceRecordReaderDataSetIterator(
      testFeatures,
      testLabels,
      minibatch,
      numLabelClasses,
      false,
      SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)
    testData.setPreProcessor(normalizer)




// ###### Network Setup ######

// ----- Configure the network -----
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .learningRate(0.005)
      .gradientNormalization(
        GradientNormalization.ClipElementWiseAbsoluteValue)
      .gradientNormalizationThreshold(0.5)
      .list()
      .layer(
        0,
        new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
      .layer(1,
             new RnnOutputLayer.Builder(LossFunction.MCXENT)
               .activation(Activation.SOFTMAX)
               .nIn(10)
               .nOut(numLabelClasses)
               .build())
      .pretrain(false)
      .backprop(true)
      .build()
    val net: MultiLayerNetwork = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(20))




    // ###### Train Network  ######
// ----- Train the network, evaluating the test set performance at each step -----
    val nEpochs: Int = 2
    val str: String =
      "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f"
    for (i <- 0 until nEpochs) {
      net.fit(trainData)
//Evaluate on the test set:
      val evaluation: Evaluation = net.evaluate(testData)
      println(String.format(str, i, evaluation.accuracy(), evaluation.f1()))
      testData.reset()
      trainData.reset()
    }




